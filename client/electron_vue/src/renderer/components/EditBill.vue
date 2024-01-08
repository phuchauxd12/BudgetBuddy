<template>
    <div class="form">
      <fwb-modal v-if="isDisplay" @close="closeModal">
        <template #header>
          <div class="flex items-center text-lg p-4">
            Edit Bill
          </div>
        </template>
        <template #body>
          <div>
            <fwb-input
              v-model="billname"
              placeholder="enter your bill name"
              label="Name"
              class="p-2"
            />
            <fwb-input
              v-model="amount"
              placeholder="$1100"
              label="Amount"
              class="p-2"
            />
            <fwb-input
              v-model="time"
              placeholder="2024-01-20"
              label="Time"
              class="p-2"
            />
            <fwb-radio v-model="picked" name="yes" label="Yes" value="true" />
            <fwb-radio v-model="picked" name="no" label="No" value="false" />
            <div class="flex justify-between p-2">
              <fwb-button @click="closeModal" color="alternative">
                Cancel
              </fwb-button>
              <fwb-button @click="addBill" color="alternative">
                Update
              </fwb-button>
            </div>
          </div>
        </template>
      </fwb-modal>
    </div>
  </template>
  
  <script setup lang="ts">
  import { ref, defineEmits, onMounted, toRefs } from 'vue';
  import { FwbButton, FwbModal, FwbInput, FwbRadio } from 'flowbite-vue';
  import axios from 'axios';
  
  const user_id = '657deedb53a90ee98e224654';
  let { data } = defineProps(['data']);
  let { bill_id, bill_name, bill_value, recurrent_reminder, recurrent_date_value } = toRefs(data);
  let id = ref(bill_id.value);
  let billname = ref(bill_name.value);
  let amount = ref(bill_value.value);
  let time = ref(recurrent_date_value.value);
  let picked = ref(recurrent_reminder.value);
  
  const isDisplay = ref(true);
  const emits = defineEmits(['close-modal']);
  
  function closeModal() {
    isDisplay.value = false;
    emits('close-modal');
  }
  
  async function addBill() {
    // Convert string inputs to numbers using parseFloat
    const billAmount = parseFloat(amount.value);
  
    // Create the request payload
    const requestData = {
      bill_name: billname.value,
      bill_value: billAmount,
      recurrent_reminder: picked.value,
      recurrent_date_value: time.value,

    };
  
    // Log the data before making the request
    console.log('Sending data to server:', requestData);
    closeModal();
  
    // Make the HTTP request
    try {
      const response = await axios.put(`http://localhost:8080/api/v1/user_bill/${user_id}/${id.value.toString()}/update`, requestData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      withCredentials: true, // Include credentials in the request
      });
  
      console.log('Server response:', response.data);
  
      // Handle success or update UI accordingly
    } catch (error) {
      console.error('Error:', error.response.data);
  
      // Handle error or update UI accordingly
    }
  }
  onMounted(() => {
    addBill();
  });
  </script>
  
  <style scoped>
    /* Your component styles go here */
  </style>