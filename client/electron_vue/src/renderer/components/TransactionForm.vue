<script setup lang="ts">
import { ref, defineEmits, onMounted, computed , watch} from 'vue';
import axios from 'axios';
const user_id = localStorage.getItem('userId') ?? '';
const userAccounts = ref([]);
const selectedAccount = ref('');


const transactionName = ref('');

const payee = ref('');
const transactionDate = ref('');
const amount = ref('');
const accountType = ref('Investment');
const transactionType = ref('Income');
const category = ref('');

const fetchBankAccount = async () => {
  try {
    console.log(accountType.value)
    const response = await axios.get(`http://localhost:8080/api/v1/bank_account/${user_id}/${accountType.value}`);
    userAccounts.value = response.data
    console.log(userAccounts.value)
  } catch (error) {
    console.error('Error fetching account information:', error);
  }
};

watch(accountType, () => {
  fetchBankAccount();
});
onMounted(fetchBankAccount)




const sendTransactionData = async () => {
  try {
    const formData = new FormData();
    formData.append('transaction_name', transactionName.value);
    formData.append('Payee', payee.value);
    formData.append('transaction_date', transactionDate.value);
    formData.append('Amount', amount.value);
    formData.append('account_name', selectedAccount.value);
    formData.append('account_type', 'Investment');
    formData.append('transaction_type', transactionType.value);
    formData.append('category', category.value);
    console.log(transactionName.value, transactionType.value, payee.value, amount.value, selectedAccount.value, accountType.value,  category.value);

    const response = await axios.post(`http://localhost:8080/api/v1/transaction/${user_id}/create`, formData, {
      withCredentials: true,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    console.log(response.data)
    alert('Transaction created successfully!')
    // Handle success or provide user feedback
  } catch (error) {
    console.error('Error creating transaction:', error);
    // Handle error or provide user feedback
  }
};
</script>

<template>

  <form  @submit.prevent="sendTransactionData" class="bg-white rounded-lg flex flex-row flex-wrap w-96 overflow-y-scroll h-96 justify-between pe-4">
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="accountType">
        Account Type
      </label>
      <div class="flex  justify-between">
        <div class="pe-8"><input type="radio" id="Income" name="accountType" value="Investment" checked  v-model="accountType">
<label class="ps-4" for="html">Investment</label><br></div>
        <div class=""><input type="radio" id="Outcome" name="accountType" value="Saving"  v-model="accountType">
<label  class="ps-4" for="Outcome">Saving</label><br></div>
<div class=""><input type="radio" id="Outcome" name="transactaccountTypeionType" value="Expense"  v-model="accountType">
<label  class="ps-4" for="Outcome">Expense</label><br></div>
</div>    
</div>
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="account">
        Account
      </label>
      <select v-model="selectedAccount" class="shadow appearance-none rounded-lg w-40 py-2 px-3 text-white mb-3 leading-tight focus:outline-none focus:shadow-outline">
        <option v-for="account in userAccounts" :key="account" :value="account.account_name">{{ account.account_name }}</option>
      </select> </div>
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="data">
        Date
      </label>
      <input required class=" shadow appearance-none  rounded-lg  w-40 py-2 px-3 text-white mb-3 leading-tight focus:outline-none focus:shadow-outline"  type="date" v-model="transactionDate">
     
    </div>
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="payee">
        Payee
      </label>
      <input v-model="payee" required class="shadow appearance-none border  rounded-lg  w-40 py-2 px-3 text-white leading-tight focus:outline-none focus:shadow-outline" id="payee" type="text" placeholder="Payee">
    </div>
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="categories">
        Categories
      </label>
      <input required class="shadow appearance-none border  rounded-lg  w-40 py-2 px-3  text-white leading-tight focus:outline-none focus:shadow-outline" id="category" type="text" placeholder="Category"  v-model="category">
    </div>
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="transactionName">
        Transaction Name
      </label>
      <input required class="shadow appearance-none border  rounded-lg  w-40 py-2 px-3 text-white leading-tight focus:outline-none focus:shadow-outline" id="acount" type="text"  v-model="transactionName" placeholder="Name">
    </div>
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="amount">
        Amount
      </label>
      <input  v-model="amount" required class="shadow appearance-none border  rounded-lg  w-40 py-2 px-3  text-white leading-tight focus:outline-none focus:shadow-outline" id="acount" type="text" placeholder="Amount">
    </div>
    <div class="mb-4">
      <label class="block text-gray-700 text-sm font-bold mb-2" for="amount">
        Transaction Type
      </label>
      <div class="flex  justify-between">
        <div class="pe-8"><input type="radio" id="Income" name="transactionType" value="Income" checked  v-model="transactionType">
<label class="ps-4" for="html">Income</label><br></div>
        <div class=""><input type="radio" id="Outcome" name="transactionType" value="Outcome"  v-model="transactionType">
<label  class="ps-4" for="Outcome">Outcome</label><br></div>

</div>    
</div>

   
    <div class="flex items-center justify-between w-full">
      <button class="w-1/2 bg-indigo-700 hover:bg-indigo-950 text-white font-bold py-2 px-4  rounded-lg  focus:outline-none focus:shadow-outline" type="submit">
       Create Transaction
      </button>
    </div>
  </form>
 
</template>

<!-- <script setup lang="ts"></script>
<template>
    <p>Hello how </p>
</template> --> 