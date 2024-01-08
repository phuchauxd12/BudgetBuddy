
<script setup lang="ts">
import axios from 'axios';
import { ref, reactive} from 'vue';
import { useRoute, useRouter } from 'vue-router';

const router = useRouter();

const state = reactive({
  full_name: ref(''),
  user_name: ref(''),
  password: ref(''),
  repassword: ref(''),
  address: ref(''),
  number: ref(''),
  email: ref(''),
});

const displayErrorMessage = (message: string, inputName: string) => {
  const inputElement = document.querySelector(`[name=${inputName}]`);
  console.log(inputElement)
  if (inputElement) {
    inputElement.style.border = '2px solid pink';
    
    const errorMessage = document.createElement('p');
    errorMessage.className = 'error-message';
    errorMessage.textContent = message;
    errorMessage.setAttribute('name', inputName); // Set the name attribute
    inputElement.parentNode.insertBefore(errorMessage, inputElement.nextSibling);
  }
};

const clearErrorStyles = () => {
  for (const field in state) {
    const inputElement = document.querySelector(`[name=${field}]`);
    if (inputElement) {
      inputElement.style.border = '1px solid #ccc';
      console.log('Cleared')
    }

    const errorMessage = document.querySelector(`p.error-message[name=${field}]`);
    if (errorMessage) {
      errorMessage.remove();
      console.log('errror msg removed')
    }
  }
};
const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};
const validateNumber = (number) => {
  const numberRegex = /^[\d\-() ]+$/;
  return numberRegex.test(number);
};



  
const register = async () => {
  const errors = []

  console.log(state.email)
clearErrorStyles()
  // Validate email, number, and password match


  // Check if there are any validation errors
 if (!validateNumber(state.number)) {
  errors.push("Invalid Phone Numner");
  displayErrorMessage(
            "Invalid Phone Number",
            "number"
          );

 }
 if (!validateEmail(state.email)) {
  errors.push("Invalid Email");
  displayErrorMessage(
            "Enter a valid email",
            "email"
          );

 }
 if (state.password != state.repassword) {
          errors.push("Passwords do not match");
          displayErrorMessage("Password does not match", "repassword");
        }

        if (errors.length > 0) {
          return;
        }

  // Create a FormData object to send form data
  const formData = new FormData();
  formData.append('user_name', state.user_name);
  formData.append('password',state.password);
  formData.append('full_name', state.full_name);
  formData.append('number', state.number);
  formData.append('email', state.email);
formData.append('address', state.address);

// console.log(state.user_name,state.password, state.full_name ,state.number, state.email,state.address)

  try {
    const response = await axios.post('http://localhost:8080/api/v1/register', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    withCredentials: true, // Include credentials in the request
  });

    // Handle successful registration
    console.log('Registration successful!', response.data);
    alert('Registration Successful!. Please sign in with the authentication key sent in your email account.')
    router.push({ name: 'Login' });
  } catch (error) {
    if (error.response && error.response.status === 409) {
      alert('Email account or username already exist!')}
    // Handle registration error
    console.error('Registration failed:', error.message, );
  }
};
</script>
<template>
    <div >
      <div class="background"></div>
      <div class="welcome-text">
        <h1>Join BudgetBuddy Today</h1>
        <p>Unlock the power of financial management</p>
      </div>
  
      <form @submit.prevent="register">
        <h3>Register Here</h3>
  <div class="grid grid-cols-2">
    <div class="form-input"><label for="user_name">Username *</label>
        <input
          required
          type="text"
          placeholder="Username"
          id="user_name"
          name="user_name"
          v-model="state.user_name"
        /></div>
        <div class="form-input"><label for="full_name">Full Name *</label>
      <input
        required
        type="text"
        placeholder="Full Name"
        id="full_name"
        v-model="state.full_name"
      /></div> 
        
      
        
        <div class="form-input">  <label for="number">Phone Number</label>
      <input
       name="number"
        type="text"
        placeholder="Phone Number"
        id="number"
        v-model="state.number"
      />
      <!-- <span v-if="state.number.error" class="error-message">{{ state.number.error }}</span> -->
</div>
<div class="form-input"><label for="address">Address</label>
      <input
       name="address"
        type="text"
        placeholder="Address"
        id="address"
        v-model="state.address"
      /> </div>
      
      <div class="form-input">  <label for="password">Password *</label>
        <input
          required
          type="password"
          placeholder="Password"
          id="password"
          name="password"
          v-model="state.password"
        />
  </div> 
   <div class="form-input"> <label for="repassword">Re-enter Password *</label>
        <input
          required
          type="password"
          placeholder="Re-enter Password"
          id="repassword"
          name="repassword"
          v-model="state.repassword"
        /> 
        <!-- <span v-if="state.repassword.error" class="error-message">{{ state.repassword.error }}</span> -->
      </div>
        <div class="form-input  col-span-2"><label for="email">Email *</label>
        <input
          required
          type=""
          placeholder="Email"
          id="email"
          name="email"
          v-model="state.email"
        />
        <!-- <span v-if="state.email.error" class="error-message">{{ state.email.error }}</span> -->
        <p class="text-sm italic pt-3"> Enter valid email to recieve authentication key </p>
      </div>
       
  </div>
  
        <button class="mt-5 text-center justify-center" type="submit">Register</button>
        <p class="m-auto mt-3 text-center fst-italic">
          <router-link to="/login" class="text-decoration-none">
            Already have an account? Sign in!
          </router-link>
        </p>
      </form>
    </div>
  </template>

  
  <style scoped>
  /* ... your existing styles ... */
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;600&display=swap');
body {
    height: 100vh;
    width: 100vw;
    background-image: url("/budgetBuddyLogin.jpg");
    background-size: cover;
    background-position: 30% center;
  }
  
  /* ... your existing styles ... */
  
  .welcome-text {
    color: #e5e5e5;
    position: absolute;
    transform: translate(-10%, -50%);
    top: 50%;
    left: 10%;
    width: 520px;
    word-wrap: break-word;
    text-align: start;
  }
  
  h1 {
    font-size: 5em;
    background: -webkit-linear-gradient(#eee, #5ed7ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  
  form {
    height: max-content;
    width: 700px;
    background-color: rgba(107, 95, 197, 0.5);
    position: absolute;
    transform: translate(-80%, -50%);
    top: 50%;
    left: 80%;
    border-radius: 10px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 40px rgba(8, 7, 16, 0.6);
    padding: 40px 35px;
  }
  form * {
    font-family: "Poppins", sans-serif;
    color: #ffffff;
    letter-spacing: 0.5px;
    outline: none;
    border: none;
  }
  
  form h3 {
    font-size: 32px;
    font-weight: 500;
    line-height: 42px;
    text-align: center;
  }
 
  label {
    display: block;
    margin-top: 30px;
    font-size: 16px;
    font-weight: 500;
    text-align: start;
  }
  
  input {
    display: block;
    height: 50px;
    width: 95%;
    background-color: rgba(255, 255, 255, 0.07);
    border-radius: 3px;
    padding: 0 10px;
    margin-top: 8px;
    font-size: 14px;
    font-weight: 300;
  }
  
  ::placeholder {
    color: #e5e5e5;
  }
  
  button {
    
    width: 100%;
    background-color: #ffffff;
    color: #080710;
    padding: 15px 0;
    font-size: 18px;
    font-weight: 600;
    border-radius: 5px;
    cursor: pointer;
  
  }
  
  .social {
    margin-top: 30px;
    display: flex;
  }
  
  .social div {
    width: 150px;
    border-radius: 3px;
    padding: 5px 10px 10px 5px;
    background-color: rgba(255, 255, 255, 0.27);
    color: #eaf0fb;
    text-align: center;
  }
  
  .social div:hover {
    background-color: rgba(255, 255, 255, 0.47);
  }
  
  .social .fb {
    margin-left: 25px;
  }
  
  .social i {
    margin-right: 4px;
  }
  
  a:hover {
    color: #ffffff;
  }
  
  a {
    color: #e5e5e5;
  }
  .error-border {
  border: 1px solid red;
}

.error-message {
  color: red;
  font-size: 14px;
  margin-top: 5px;
  display: block;
}


  </style>