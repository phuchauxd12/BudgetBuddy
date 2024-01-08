<script setup lang="ts">
import { onMounted, ref } from 'vue';
import axios from 'axios';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../../@/components/ui/card';
import { Separator } from '../../@/components/ui/separator'

const billData = ref([]);
const showReminderOptions = ref(false);
const selectedBillId = ref('');
const user_id = localStorage.getItem('userId') ?? '';

const findBillById = (billId: string) => {
  return billData.value.find((bill) => bill.bill_id === billId);
};

const toggleReminderOptions = (billId: string) => {
  showReminderOptions.value = !showReminderOptions.value;
  selectedBillId.value = billId;
};

const turnOffReminder = async () => {
  try {
    
    const formData = new FormData();
   
    const selectedBill = findBillById(selectedBillId.value);
    console.log(selectedBill)
    // Assuming you have user_id and bill_id values

   
    const bill_id = selectedBillId.value;
   
    formData.append('bill_name', selectedBill.bill_name);
    formData.append('bill_value', selectedBill.bill_value);
   
    formData.append('recurrent_reminder', false);
    formData.append('recurrent_date_value', selectedBill.recurrent_date_value);
  
    // Send a PUT request to update the bill
    const response = await axios.put(`http://localhost:8080/api/v1/user_bill/${user_id}/${bill_id}/update`, formData, {
      withCredentials: true,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });


    // Handle the response or provide user feedback
    console.log(response.data);

    // Optionally, you can close the reminder options after turning it off
    showReminderOptions.value = false;
    location.reload();

  } catch (error) {
    console.error('Error updating bill:', error);
    // Handle error or provide user feedback
  }
};

onMounted(async () => {
  try {
    const response = await axios.get('http://localhost:8080/api/v1/user_bill/6593ccdf025b256e0ffe24e8/sorted');
    billData.value = response.data;
    console.log('bill data:', billData.value);
  } catch (error) {
    console.error('Error fetching bill:', error);
  }
});
</script>

<template>
  <div>
    <h1 class="font-semibold text-lg text-left ps-5 pb-5">Upcoming Bills</h1>
   
    <div class="  justify-center gap-4 h-44 carousel rounded-box w-full">
    <div class="carousel-item w-full bg-indigo-700">
        <!-- Use v-for to loop through a range (0 to 3) and create a card for each iteration -->
      <Card
        v-for="(bill, index) in billData"
        :key="index"
        class="w-48  h-36 mt-3 me-4 relative overflow-hidden ms-4 "
       
       
      >
      <div class="relative">
      <button  @click="toggleReminderOptions(bill.bill_id)" class="text-xl p-0 text-right w-full ps-40 font-bold" >...</button>
      <div @click="turnOffReminder" v-if="showReminderOptions && selectedBillId === bill.bill_id" class="z-40 bg-indigo-700 rounded-lg w-max px-3 py-2 absolute text-white shadow-xl top-6 right-0 " id="${bill.bill_id}"> Turn off reminder</div></div>
        <div class="relative z-10 flex flex-col" v-if="bill.recurrent_reminder==true" > <!-- Place text inside a relative container -->
           
            <CardHeader class=" text-ellipsis text-xl font-semibold text-left h:1/2 text-indigo-800 px-4 py-0">   
                {{bill.bill_name}}
                <Separator/>
            <CardTitle class="text-black font-bolder text-sm overflow-hidden text-ellipsis hover:underline ">
              <a :href="bill.bill_url" target="_blank" rel="noopener noreferrer" class="p-0">{{bill.recurrent_date_value}}  </a>
            </CardTitle>
            <CardDescription class="text-black text-lg">
           $ {{bill.bill_value}}
            </CardDescription>
        </CardHeader>
     
         
        </div>
      </Card>
    </div>
    </div>
   
  </div>
</template>
