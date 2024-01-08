<template>
    <div class="form">
      <fwb-modal v-if="isDisplay" @close="closeModal">
        <template #header>
          <div class="flex items-center text-lg p-4">
            Add Subscription
          </div>
        </template>
        <template #body>
          <div>
            <fwb-input
              v-model="subname"
              placeholder="enter your subscription name"
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
              placeholder="Jan 2024"
              label="Time"
              class="p-2"
            />
            <div class="flex justify-between p-2">
              <fwb-button @click="closeModal" color="alternative">
                Cancel
              </fwb-button>
              <fwb-button @click="addSub" color="alternative">
                Add
              </fwb-button>
            </div>
          </div>
        </template>
      </fwb-modal>
    </div>
  </template>
  
  <script setup lang="ts">
  import { ref, defineEmits, onMounted } from 'vue';
  import { FwbButton, FwbModal, FwbInput, FwbSelect } from 'flowbite-vue';
  import axios from 'axios';
  
  const user_id = '657deedb53a90ee98e224654';
  const subname = ref('');
  const amount = ref('');
  const time = ref('');
  const type = ref('Subscription')
  
  const isDisplay = ref(true);
  const emits = defineEmits(['close-modal']);
  
  function closeModal() {
    isDisplay.value = false;
    emits('close-modal');
  }
  onMounted(() => {
    addSub();
  });
  async function addSub() {
    // Convert string inputs to numbers using parseFloat
    const currentAmount = parseFloat(amount.value);
  
    // Create the request payload
    const requestData = {
      spending_name: subname.value,
      spending_type: type.value,
      amount: currentAmount,
      time_duration: time.value,
    };
  
    // Log the data before making the request
    console.log('Sending data to server:', requestData);
  
    // Make the HTTP request
    try {
      const response = await axios.post(`http://localhost:8080/api/v1/plan_spending/${user_id}/create`, requestData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      withCredentials: true, // Include credentials in the request
      });
  
      console.log('Server response:', response.data);
      closeModal();
  
      // Handle success or update UI accordingly
    } catch (error) {
      console.error('Error:', error.response.data);
  
      // Handle error or update UI accordingly
    }
  }
  </script>
  
  <style scoped>
    /* Your component styles go here */
  </style>