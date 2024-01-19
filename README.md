# BudgetBuddy

BudgetBuddy is a financial management app that help you manage your financial.
We provide a desktop application that could:
- Save transaction history
- Save money toward goal
- Planning a head for your month
- Predict your future expense
- Stock and Investment News Real-time updated

## Folder Structure
```
\app: source code for the back-end
\client: source code for the front-end
```
## BudgetBuddy Deployment

### Step 1: Run Backend (run command from root folder)

 1: Create a `local.env` file 
File containa secret value like MongoDB URI, API key, etc. inclued create email.env in models, which has email and email app password
example:
   FROM_EMAIL=doannguyenphuchau@gmail.com
   EMAIL_PASSWORD=ucee mmkg qwxy avyp
   
 MONGODB_URI=mongodb+srv://phuchauxd12:Abcd0123@cluster0.lf8sh9p.mongodb.net/
 
 2: Install packages 
```pip install -r requirements-dev.txt```

 3: Deploy backend 
```python app/main.py``` or ```python3 app/main```

 4: Back end running on port 8080 
Go to 'http://localhost:8080/api/v1/docs' to see APIs


### Step 2: Run Frontend (run command from root folder)

 1: Go to Client Folder
```cd client```

 2: Go to electron_vue Folder 
```cd electron_vue```

 3: Install dependencies 
```npm i```

 4: Run the environment 
```npm run dev```


#### Note
- Make sure you have installed nodejs version 18. or higher, python/python3 and pip.
- If you encounter problem related vue version when deploy frontend, run “npm vue update”

  
### Step 3: App is ready and running on Electron application or visit localhost:8081/
