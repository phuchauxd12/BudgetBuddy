
<template>
  <section class="networth">
    <div class="total">
      <p class="label">Networth</p>
      <h2 class="label">{{ totalNetworth }}</h2>
    </div>
    <div class="networth_container">
      <article class="bank_acc">
        <div v-for="accountType in uniqueAccountTypes" :key="accountType">
          <button @click="toggleAccountType(accountType)">
            <p>{{ accountType }}</p>
            <p>{{ getSumOfAccount(accountType) }}</p>
          </button>
          <div v-if="selectedAccountTypes.includes(accountType)">
            <div v-for="account in getAccountsByType(accountType)" :key="account[0]" class="flex justify-between">
              <p class="flex-1 pb-2.5">{{ account[0] }}</p>
              <p class="flex-1 pb-2.5">{{ account[1] }}</p>
            </div>
          </div>
        </div>
      </article>

      <button @click="addAccount" class="add">ADD</button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, defineEmits, onMounted, computed } from 'vue';
import axios from 'axios';

const user_id = '6593ccdf025b256e0ffe24e8';
const userAccounts = ref([]);
const selectedAccountTypes = ref([]);

const fetchBankAccount = async () => {
  try {
    const response = await axios.get(`http://localhost:8080/api/v1/${user_id}`);
    userAccounts.value = response.data.list_account_name;
  } catch (error) {
    console.error('Error fetching account information:', error);
  }
};

onMounted(() => {
  fetchBankAccount();
});

const emits = defineEmits(['add-account']);

const addAccount = () => {
  emits('add-account');
};

const uniqueAccountTypes = computed(() => {
  return Array.from(new Set(userAccounts.value.map(account => account[4])));
});

const getSumOfAccount = (accountType) => {
  return userAccounts.value
    .filter(account => account[4] === accountType)
    .reduce((sum, account) => sum + parseFloat(account[1]), 0)
    .toFixed(2);
};

const getAccountsByType = (accountType: any) => {
  return userAccounts.value.filter(account => account[4] === accountType);
};

const totalNetworth = computed(() => {
  return userAccounts.value.reduce((sum, account) => sum + parseFloat(account[1]), 0).toFixed(2);
});

const toggleAccountType = (accountType: any) => {
  const index = selectedAccountTypes.value.indexOf(accountType);
  if (index === -1) {
    selectedAccountTypes.value.push(accountType);
  } else {
    selectedAccountTypes.value.splice(index, 1);
  }
};
</script>

<style scoped>
  .notShown {
    visibility: hidden !important;
  }

  button {
    background-color: #6A56E7;
    width: 250px;
    border: none;
    border-radius: 20px;
    padding: 8px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
  }
  .add{
  color: white;
  font-weight: 700;
  font-size: small;
  position: absolute;
  bottom: 8px;
  left: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
