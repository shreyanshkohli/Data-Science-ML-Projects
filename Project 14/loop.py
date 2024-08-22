from sklearn.model_selection import train_test_split

def optimise(model, x, y):
    best_score = -float('inf')
    best_index = -1
    
    for i in range(1, 100):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=i)
        model.fit(xtrain, ytrain)
        score = model.score(xtest, ytest)
        
        if score > best_score:
            best_score = score
            best_index = i
    
    return best_score, best_index