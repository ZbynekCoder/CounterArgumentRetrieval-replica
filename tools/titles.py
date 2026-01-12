import os

training = 'data/training/'
validation = 'data/validation/'
test = 'data/test/'

def collect_titles(path):
    domains = os.listdir(path)
    titles = []
    for x in domains:
        if(x[:2]=='__' or x[0]=='.'):
            continue
        temp_dir = path + x +'/'
        for j in os.listdir(temp_dir):
            titles.append(j)
    return titles


if __name__ == '__main__':

    training_titles = collect_titles(training)
    print(len(training_titles))
    validation_titles = collect_titles(validation)
    print(len(validation_titles))
    test_titles = collect_titles(test)
    print(len(test_titles))
    print(len(set(training_titles + validation_titles)))
    print(len(set(training_titles + test_titles)))

