<template>
  <div class="plan-income">
    <article class="income">
      <div @click="toggleShow" class="display">
        <div class="aqq">
          <p>Income</p>
          <p class="aqq2" v-if="isShow" @click="toggleaddIncome">+ Add income</p>
        </div>
        <div class="gia"><p class="gia1">+{{ sumAmount(userIncome).toFixed(2) }}</p></div>
      </div>
      <div v-for="incomeEntry in userIncome" class="bg-white flex justify-between rounded-lg p-1.5 m-2" v-if="isShow">
        <p class="flex-1 pb-1.5">{{ incomeEntry.spending_name }}</p>
        <p class="flex-1 pb-1.5">{{ incomeEntry.amount.toFixed(2) }}</p>
        <button @click="clickEdit(incomeEntry)">Edit</button>
        <button @click="clickDelete(incomeEntry.id)">Delete</button>
      </div>
    </article>
    <article class="income">
      <div @click="toggleShow3" class="display">
        <div class="aqq">
          <p>Bill</p>
          <p class="aqq2" v-if="isShow3" @click="toggleaddBill">+ Add Bill</p>
        </div>
        <div class="gia"><p class="gia1">-{{ sumBill(userBill).toFixed(2) }}</p></div>
      </div>
      <div v-for="billEntry in userBill" class="bg-white flex justify-between rounded-lg p-1.5 m-2" v-if="isShow3">
        <p class="flex-1 pb-1.5">{{ billEntry.recurrent_date_value}}</p>
        <p class="flex-1 pb-1.5">{{ billEntry.bill_name }}</p>
        <p class="flex-1 pb-1.5">{{ billEntry.bill_value.toFixed(2) }}</p>
        <p class="flex-1 pb-1.5">{{ billEntry.recurrent_reminder}}</p>
        <button @click="clickEdit3(billEntry)">Edit</button>
        <button @click="clickDelete2(billEntry.bill_id)">Delete</button>
      </div>
    </article>
    <article class="income">
      <div @click="toggleShow2" class="display">
        <div class="aqq">
          <p>Subscription</p>
          <p class="aqq2" v-if="isShow2" @click="toggleaddSub">+ Add subscription</p>
        </div>
        <div class="gia"><p class="gia1">-{{ sumAmount(userSubscription).toFixed(2) }}</p></div>
      </div>
      <div v-for="subEntry in userSubscription" class="bg-white flex justify-between rounded-lg p-1.5 m-2" v-if="isShow2">
        <p class="flex-1 pb-1.5">{{ subEntry.spending_name }}</p>
        <p class="flex-1 pb-1.5">{{ subEntry.amount.toFixed(2) }}</p>
        <button @click="clickEdit2(subEntry)">Edit</button>
        <button @click="clickDelete(subEntry.id)">Delete</button>
      </div>
    </article>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, toRefs, watch, watchEffect } from 'vue';
import axios from 'axios';

const user_id = '6593ccdf025b256e0ffe24e8';
const userIncome = ref([]);
const userSubscription = ref([]);
const userBill = ref([]);
const isShow = ref(false);
const isShow2 = ref(false);
const isShow3 = ref(false);
const { month, year }  = defineProps(['month', 'year']);
let that_month = ref(month);
let that_year = ref(year);
let editData = ref(null)
let deleteId = ref(null)

watchEffect(async () => {
  console.log('newmonth',that_month.value)
  
})

const fetchPlan = async () => {
  try {
    const response = await axios.get(`http://localhost:8080/api/v1/plan_spending/${user_id}/Jan 2024`);
    userIncome.value = response.data.filter((entry: { spending_type: string; }) => entry.spending_type === 'Income');
    userSubscription.value = response.data.filter((entry: { spending_type: string; }) => entry.spending_type === 'Subscription');
    console.log('Data from the server:', userIncome.value);
    console.log('Data from the server2:', userSubscription.value);

  } catch (error) {
    console.error('Error fetching account information:', error);
  }
};
const fetchBill = async () => {
  try {
    const response = await axios.get(`http://localhost:8080/api/v1/user_bill/${user_id}`);
    userBill.value = response.data
    console.log('Data from the server3:', userBill.value);
  } catch (error) {
    console.error('Error fetching account information:', error);
  }
};

const Delete = async (deleteId) => {
  try {
    const response = await axios.delete(`http://localhost:8080/api/v1/plan_spending/${user_id}/${deleteId}`);
    console.log('Delete this ID', response.data);
    await fetchPlan();
    await fetchBill();
  } catch (error) {
    console.error('Error deleting entry:', error);
  }
};
const Delete2 = async (deleteId) => {
  try {
    const response = await axios.delete(`http://localhost:8080/api/v1/user_bill/${user_id}/${deleteId}/delete`);
    console.log('Delete this ID', response.data);
    await fetchPlan();
    await fetchBill();
  } catch (error) {
    console.error('Error deleting entry:', error);
  }
};

const toggleShow = () => {
  isShow.value = !isShow.value;
};
const toggleShow2 = () => {
  isShow2.value = !isShow2.value;
};
const toggleShow3 = () => {
  isShow3.value = !isShow3.value;
};

const sumAmount = (entries: any[]) => {
  return entries.reduce((total: any, entry: { amount: any; }) => total + entry.amount, 0);
};
const sumBill = (entries: any[]) => {
  return entries.reduce((total: any, entry: { bill_value: any; }) => total + entry.bill_value, 0);
};

onMounted(() => {
  fetchPlan();
  fetchBill();
});

const clickDelete = (itemDelete: any) => {
  deleteId = itemDelete;
  Delete(deleteId);
};
const clickDelete2 = (itemDelete: any) => {
  deleteId = itemDelete;
  Delete2(deleteId);
};

const clickEdit = (itemEdit: any) => {
  editData = itemEdit,
  emits("edit-income", editData);
  console.log("edit-income:",editData)
}

const clickEdit2 = (itemEdit: any) => {
  editData = itemEdit,
  emits("edit-sub", editData);
  console.log("edit-sub:",editData)
}
const clickEdit3 = (itemEdit: any) => {
  editData = itemEdit,
  emits("edit-bill", editData);
  console.log("edit-bill:",editData)
}

const emits = defineEmits(['add-income', 'add-sub', 'add-bill','edit-income', 'edit-sub', 'edit-bill']);

const toggleaddIncome = () => {
  emits('add-income');
};

const toggleaddSub = () => {
  emits('add-sub');
};
const toggleaddBill = () => {
  emits('add-bill');
};
</script>



<style scoped>
  .plan-income{ 
      position: absolute;
      top: 20px;
      left: 20px;
      width: 93%;
  }
  .income{
    margin-top: 30px;
    background-color: lavender;
    border-radius: 10px;
    padding: 10px;
  }
  .aqq{
    display: flex;
  }
  .aqq2{
    margin-left: 10px;
    color:rgb(46, 119, 255);
  }
  .gia1{
    margin-left: 10px;
  }
  .display{
    border: none;
    width: 100%;
    display: flex;
    padding: 7px;
    justify-content: space-between;
    cursor: pointer;
  }
</style>