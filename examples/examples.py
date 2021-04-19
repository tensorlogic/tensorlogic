import sparse as sp
import numpy as np
import torch.utils.data
import torchvision

from structs import QueryData
from program import Program


def friends_example():

    # get some random data over a domain of 50 people:
    # - a sparse matrix of a priori friends relations.
    # - a sparse matrix of a priori stressed people.
    # - a sparse matrix of people who are known a priori to smoke.
    # - a sparse matrix of people who are known a priori to drink.
    friends_init = (sp.random(density=0.15, shape=(50, 50)) > 0).astype(np.float32)
    stressed_init = (sp.random(density=0.45, shape=(50,)) > 0).astype(np.float32)
    smokes_init = (sp.random(density=0.2, shape=(50,)) > 0).astype(np.float32)
    drinks_init = (sp.random(density=0.25, shape=(50,)) > 0).astype(np.float32)

    # compute the following rules using the sparse matrices:
    # - friends of friends are friends.
    # - people who are stressed are either known a priori to be stressed or are friends with stressed people.
    # - people who smoke are either known a priori to smoke, or are friends with people who smoke, or are stressed.
    # - people who drink are either known a priori to drink, or are friends with people who drink, or are stressed.
    # - people who have cancer are either smokers, drinkers, or are stressed, but each factor contributes by a certain
    # weight.

    friends = friends_init @ friends_init
    stressed = sp.dot(friends, stressed_init) + stressed_init
    smokes = sp.dot(friends, smokes_init) + smokes_init + stressed
    drinks = sp.dot(friends, drinks_init) + drinks_init + stressed

    smokes_weight, drinks_weight, stressed_weight = 0.5, 0.3, 0.2
    cancer = smokes_weight * smokes + drinks_weight * drinks + stressed_weight * stressed

    # now we compute the people who are friends and share one of the attributes of having cancer, drinking, smoking or
    # being stressed.
    cancer_friends = friends * cancer.reshape((50, 1)) * cancer.reshape((1, 50))
    drinks_friends = friends * drinks.reshape((50, 1)) * drinks.reshape((1, 50))
    smokes_friends = friends * smokes.reshape((50, 1)) * smokes.reshape((1, 50))
    stressed_friends = friends * stressed.reshape((50, 1)) * stressed.reshape((1, 50))

    # create a TensorLogic Program which encodes the rules above as:

    # Friends[x, y] :- Friends_init[x, z]Friends_init[z, y]
    #
    # Stressed[x] :- Friends[x, y]Stressed_init[y]
    # Stressed[x] :- Stressed_init[x]
    #
    # Smokes[x] :- Friends[x, y]Smokes_init[y]
    # Smokes[x] :- Smokes_init[x]
    # Smokes[x] :- Stressed[x]
    #
    # Drinks[x] :- Friends[x, y]Drinks_init[y]
    # Drinks[x] :- Drinks_init[x]
    # Drinks[x] :- Stressed[x]
    #
    # Cancer[x]: - Smokes_weight[]Smokes[x]
    # Cancer[x]: - Drinks_weight[]Drinks[x]
    # Cancer[x]: - Stressed_weight[]Stressed[x]
    #
    # CancerFriends[x, y] :- Cancer[x]Cancer[y]Friends[x, y]
    # DrinksFriends[x, y] :- Drinks[x]Drinks[y]Friends[x, y]
    # SmokesFriends[x, y] :- Smokes[x]Smokes[y]Friends[x, y]
    # StressedFriends[x, y] :- Stressed[x]Stressed[y]Friends[x, y]
    p = Program("friends.txt")

    # compile the program.
    # the ranges are given as SetRanges for stressed, smokers, drinkers and as one CoordRange for Friends.
    # the values of the tensors are set to True.
    p.compile(constant_ranges=dict(friends=friends_init.coords, stressed=stressed_init.coords.flatten(),
                                   smokers=smokes_init.coords.flatten(), drinkers=drinks_init.coords.flatten()),
              constant_tensors=dict(Drinks_init=True, Smokes_init=True, Friends_init=True, Stressed_init=True,
                                    Smokes_weight=smokes_weight, Drinks_weight=drinks_weight, Stressed_weight=stressed_weight))

    # run a couple of queries for some random people checking all of their attributes.

    # for the friend queries we check who are i's friends and the query will return a vector with 50 entries with
    # non zero values at indices j where we have evidence that j is friends with i.
    # the other queries will return a scalar indicating whether i has that property.
    for i in np.random.randint(0, 50, (20, )):

        result = p.run(queries=[QueryData(tensor_name="Friends_init", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
                                QueryData(tensor_name="Stressed_init", domain_tuple=("i",), domain_vals=dict(i=i)),
                                QueryData(tensor_name="Smokes_init", domain_tuple=("i",), domain_vals=dict(i=i)),
                                QueryData(tensor_name="Drinks_init", domain_tuple=("i",), domain_vals=dict(i=i)),
                                QueryData(tensor_name="Friends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
                                QueryData(tensor_name="Stressed", domain_tuple=("i",), domain_vals=dict(i=i)),
                                QueryData(tensor_name="Smokes", domain_tuple=("i",), domain_vals=dict(i=i)),
                                QueryData(tensor_name="Drinks", domain_tuple=("i",), domain_vals=dict(i=i)),
                                QueryData(tensor_name="Cancer", domain_tuple=("i",), domain_vals=dict(i=i))])[0]

        # check if the values computed by the program are the same as the ones computed manually.
        assert np.allclose(result[0], friends_init[i].todense())
        assert np.allclose(result[1], stressed_init[i])
        assert np.allclose(result[2], smokes_init[i])
        assert np.allclose(result[3], drinks_init[i])
        assert np.allclose(result[4], friends[i].todense())
        assert np.allclose(result[5], stressed[i])
        assert np.allclose(result[6], smokes[i])
        assert np.allclose(result[7], drinks[i])
        assert np.allclose(result[8], cancer[i])

    # pick some random pairs of people and check if the pair of friends has certain properties.
    # the first type of query checks if the pair of friends has that property. This returns a scalar.
    # the second type of query check who are i's friends who have that propery. This returns a vector of size 50.
    for i, j in zip(np.random.randint(0, 50, (20, )), np.random.randint(0, 50, (20, ))):

        result = p.run(queries=[QueryData(tensor_name="Friends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
                                QueryData(tensor_name="Friends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
                                QueryData(tensor_name="CancerFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
                                QueryData(tensor_name="CancerFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
                                QueryData(tensor_name="DrinksFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
                                QueryData(tensor_name="DrinksFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
                                QueryData(tensor_name="SmokesFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
                                QueryData(tensor_name="SmokesFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
                                QueryData(tensor_name="StressedFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
                                QueryData(tensor_name="StressedFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None))])[0]

        assert np.allclose(result[0], friends[i, j])
        assert np.allclose(result[1], friends[i, :].todense())
        assert np.allclose(result[2], cancer_friends[i, j])
        assert np.allclose(result[3], cancer_friends[i, :].todense())
        assert np.allclose(result[4], drinks_friends[i, j])
        assert np.allclose(result[5], drinks_friends[i, :].todense())
        assert np.allclose(result[6], smokes_friends[i, j])
        assert np.allclose(result[7], smokes_friends[i, :].todense())
        assert np.allclose(result[8], stressed_friends[i, j])
        assert np.allclose(result[9], stressed_friends[i, :].todense())

    # here we check if 200 random pairs of people are friends and are likely to have cancer.
    coords = np.stack([np.random.randint(0, 50, (200,)), np.random.randint(0, 50, (200,))])
    result = p.run(queries=[QueryData(tensor_name="CancerFriends", domain_tuple=("x.0", "x.1"),
                                      domain_vals=dict(x=coords))])[0]
    assert np.allclose(result[0], cancer_friends[tuple(coords)].todense())

    # here we check if the 200 people who were first in the random pair are friends with any of the first 20 people
    # and if both of them smoke.
    result = p.run(queries=[QueryData(tensor_name="SmokesFriends", domain_tuple=("x.0", "s"),
                                      domain_vals=dict(x=coords[0:1], s=slice(0, 20)))])[0]
    assert np.allclose(result[0], smokes_friends[coords[0], slice(0, 20)].todense())


def MLP_example():

    # multi-layer perceptron example

    # load the MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/.torch_data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=16, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/.torch_data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=16, shuffle=True)

    num_epochs = 10

    # compile the program stored in MLP.txt
    # note that the program does not expect any ranges as inputs. In this case, everything is treated as a DummyRange
    # and the rules apply universally.
    # furthermore, we compile the program using an ADAM optimizer
    p = Program("MLP.txt")
    p.compile(optimizer="adam", optimizer_kwargs=dict(lr=0.0001))

    # for num_epochs
    for epoch in range(0, num_epochs):

        running_loss, num_iter = 0.0, 0

        # get the data from the data_loader
        for inp, labels in train_loader:
            # create the input_tensors dictionary which sets the values of some(not necessarily all) of the
            # the program's input tensors
            # run the program. Note that we set the losses parameter to the name of the loss declared in the program
            # definition. This way the loss is computed. Then, because the backprop flag is set to True, the
            # loss will get backpropagated to the weight tensors
            input_tensors = dict(Input=inp.reshape((16, 784)), Labels=labels)
            running_loss += p.run(losses=("loss", ), input_tensors=input_tensors, backprop=True)[1][0]
            num_iter += 1

        print('Training epoch %d completed. Loss: %5f' % (epoch + 1, running_loss/num_iter))

        correct, num_examples = 0, 0

        # after one epoch is time to test the performance
        for inp, labels in test_loader:

            # again, we set the input tensors
            input_tensors = dict(Input=inp.reshape((16, 784)))

            # we set some queries to compute:
            # - the first query computes the log of the predicted probabilities of each digit
            # - the second and third queries will compute the preactivations of the odd digits, and
            # even digits respectively(for illustration purposes).
            query_all = QueryData(tensor_name="A2", domain_tuple=("batch", "x"),
                                  domain_vals=dict(batch=None, x=None))
            query_odds = QueryData(tensor_name="PA2", domain_tuple=("batch", "x"),
                                   domain_vals=dict(batch=None, x=range(1, 11, 2)))
            query_evens = QueryData(tensor_name="PA2", domain_tuple=("batch", "x"),
                                    domain_vals=dict(batch=None, x=range(0, 10, 2)))

            # run the queries, note that we are not computing the loss nor are we backpropagating
            query_all, query_odds, query_evens = p.run(queries=(query_all, query_odds, query_evens),
                                                       input_tensors=input_tensors)[0]

            # check the accuracy of the predictions
            predictions = torch.argmax(query_all, dim=1)
            correct += torch.sum(predictions == labels)
            num_examples += predictions.shape[0]

        print('Test after epoch %d completed. Test Accuracy: %5f' % (epoch + 1, correct / num_examples))


friends_example()
MLP_example()