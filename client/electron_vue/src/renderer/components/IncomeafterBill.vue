<template>
<div class="h-6"></div>
  <button class="w-6 h-6 p-0 me-4 bg-indigo-700 align-middle text-white font-bold cursor-pointer rounded-md" @click="prevMonth">&lt;</button>
    <span class="font-bold text-lg">{{ monthNames[currentMonth] }} {{ currentYear }}</span>
    <button class="w-6 h-6 p-0 me-4 bg-indigo-700 align-middle text-white font-bold ms-4 cursor-pointer rounded-md" @click="nextMonth">&gt;</button>
  <div class="plan-income mt-8">
    <article class="income">
      <div @click="toggleShow" class="display">
        <div class="aqq">
          <p>Income</p>
          <p class="aqq2" v-if="isShow" @click="toggleaddIncome(formattedMonth, currentYear)">+ Add income</p>
        </div>
        <div class="gia "><p class="gia1 font-bold text-green-700">+ ${{ sumAmount(userIncome).toFixed(2) }}</p></div>
      </div>
      <div v-for="incomeEntry in userIncome" class="bg-white flex justify-between rounded-lg p-1.5 m-2" v-if="isShow">
        <p class="flex-1 pb-1.5">{{ incomeEntry.spending_name }}</p>
        <p class="flex-1 pb-1.5">{{ incomeEntry.amount.toFixed(2) }}</p>
        <Button class="cursor-pointer w-fit h-6 px-2 me-4 bg-indigo-700 align-middle text-white" @click="clickEdit(incomeEntry)">Edit</Button>
        <Button class="cursor-pointer w-fit h-6 px-2 me-4 align-middle text-white bg-orange-800" @click="clickDelete(incomeEntry.id)">Delete</Button>
        
      </div>
    </article>
    <article class="income">
      <div @click="toggleShow3" class="display">
        <div class="aqq">
          <p>Bill</p>
          <p class="aqq2" v-if="isShow3" @click="toggleaddBill">+ Add Bill</p>
        </div>
        <div class="gia"><p class="gia1 font-bold text-orange-700">- ${{ sumBill(userBill).toFixed(2) }}</p></div>
      </div>
      <div v-for="billEntry in userBill" class="bg-white flex justify-between rounded-lg p-1.5 m-2" v-if="isShow3">
        <p class="flex-1 pb-1.5">{{ billEntry.recurrent_date_value}}</p>
        <p class="flex-1 pb-1.5">{{ billEntry.bill_name }}</p>
        <p class="flex-1 pb-1.5">{{ billEntry.bill_value.toFixed(2) }}</p>
        <p class="flex-1 pb-1.5">{{ billEntry.recurrent_reminder}}</p>
        <Button class="cursor-pointer w-fit h-6 px-2 me-4 bg-indigo-700 align-middle text-white" @click="clickEdit3(billEntry)">Edit</Button>
        <Button class="cursor-pointer w-fit h-6 px-2 me-4 bg-orange-800 align-middle text-white" @click="clickDelete2(billEntry.bill_id)">Delete</Button>
      </div>
    </article>
    <article class="income">
      <div @click="toggleShow2" class="display">
        <div class="aqq">
          <p>Subscription</p>
          <p class="aqq2" v-if="isShow2" @click="toggleaddSub(formattedMonth, currentYear)">+ Add subscription</p>
        </div>
        <div class="gia"><p class="gia1 font-bold text-orange-700">- ${{ sumAmount(userSubscription).toFixed(2) }}</p></div>
      </div>
      <div v-for="subEntry in userSubscription" class="bg-white flex justify-between rounded-lg p-1.5 m-2" v-if="isShow2">
        <p class="flex-1 pb-1.5">{{ subEntry.spending_name }}</p>
        <p class="flex-1 pb-1.5">{{ subEntry.amount.toFixed(2) }}</p>
        <Button class="cursor-pointer w-fit h-6 px-2 me-4 bg-indigo-700 align-middle text-white" @click="clickEdit2(subEntry)">Edit</Button>
        <Button class="cursor-pointer w-fit h-6 p-2 me-4 bg-orange-800 align-middle text-white" @click="clickDelete(subEntry.id)">Delete</Button>
      </div>
    </article>
  </div>
</template>

<script setup lang="ts">

import { Button } from '../../@/components/ui/button'
import { ref, onMounted, toRefs, watchEffect, watch, computed} from 'vue';
import axios from 'axios';


const  currentMonth= ref(new Date().getMonth())
const  currentYear= ref(new Date().getFullYear())
const  monthNames= ref([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ])

const  prevMonth =() => {
    if (currentMonth.value === 0) {
      currentMonth.value = 11;
      currentYear.value -= 1;
    } else {
      currentMonth.value -= 1;
    }};
const  nextMonth =()=> {
    if (currentMonth.value === 11) {
      currentMonth.value = 0;
      currentYear.value += 1;
    } else {
      currentMonth.value += 1;
    }
  };

  watch (currentMonth,async(newMonth)=>{
    console.log('newMonth', newMonth)
    console.log('year', currentYear.value)
    console.log('formatMonth', formattedMonth.value)
    fetchPlan();
    fetchBill();
    emits('send-month', newMonth);
    emits('send-year', currentYear.value);
  }
  )
  

const user_id = localStorage.getItem('userId') ?? '';
const userIncome = ref([]);
const userSubscription = ref([]);
const userBill = ref([]);
const isShow = ref(false);
const isShow2 = ref(false);
const isShow3 = ref(false);
let that_year = ref( "2024");
let editData = ref(null)
let sendMonth = ref(null)
let deleteId = ref(null) 

let formattedMonth = ref(
  computed(() => {
    return new Date(`${that_year.value}-${currentMonth.value + 1}-10`)
      .toLocaleString('en-US', { month: 'short' });
  })
);


  


const fetchPlan = async () => {
  try {
    const response = await axios.get(`http://localhost:8080/api/v1/plan_spending/${user_id}/${formattedMonth.value} ${currentYear.value}`);
    userIncome.value = response.data.filter((entry: { spending_type: string; }) => entry.spending_type === 'Income');
    userSubscription.value = response.data.filter((entry: { spending_type: string; }) => entry.spending_type === 'Subscription');
    console.log('Data from the server:', userIncome.value);
    console.log('Data from the server2:', userSubscription.value);
    const totalIncome = userIncome.value.reduce((sum, incom) => sum + incom.amount, 0);
    const totalSub = userSubscription.value.reduce((sum, sub) => sum + sub.amount, 0);
    emits("send-in", totalIncome);
    emits("send-sub", totalSub);
  } catch (error) {
    console.error('Error fetching account information:', error);
  }
};
const fetchBill = async () => {
  try {
    const response = await axios.get(`http://localhost:8080/api/v1/user_bill/${user_id}`);
    const currentMonthBills = response.data.filter((bill) => {
      const billDate = new Date(bill.recurrent_date_value);
      return billDate.getMonth() === currentMonth.value && billDate.getFullYear() === currentYear.value;
    });
    userBill.value = currentMonthBills;
    console.log('Data from the server3:', userBill.value);
    const totalBill = userBill.value.reduce((sum, bill) => sum + bill.bill_value, 0);
    emits("send-bil", totalBill);
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

const emits = defineEmits(['add-income', 'add-sub', 'add-bill','edit-income', 'edit-sub', 'edit-bill', 'send-month','send-year','send-in','send-sub','send-bil']);

const toggleaddIncome = (formattedMonth: any, currentYear: any) => {
  const sendMonth = `${formattedMonth} ${currentYear}`;
  emits('add-income', sendMonth);
};

const toggleaddSub = (formattedMonth: any, currentYear: any) => {
  const sendMonth = `${formattedMonth} ${currentYear}`;
  emits('add-sub',sendMonth);
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
      width: 100%;
     padding-left: 2rem;
      padding-right: 4rem;
  }
  .income{
    margin-top: 30px;
    height: fit-content;
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
    height:40px !important;
  }
</style>