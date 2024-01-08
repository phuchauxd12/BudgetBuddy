<template>
    <div class="form">
      <fwb-modal v-if="isDisplay" @close="closeModal">
        <template #header>
          <div class="flex items-center text-lg p-4">
            Add your goal
          </div>
        </template>
        <template #body>
          <div>
            <fwb-input
              v-model="goalname"
              placeholder="enter your goal name"
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
              placeholder="2024/01/30"
              label="Time"
              class="p-2"
            />
            <fwb-input
              v-model="linkaccount"
              placeholder="Saving 1"
              label="Linked Account *optional*"
              class="p-2"
            />
            <div class="flex justify-between p-2">
              <fwb-button @click="closeModal" color="alternative">
                Cancel
              </fwb-button>
              <fwb-button @click="addGoal" color="alternative">
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
  
  const user_id = '6593ccdf025b256e0ffe24e8';
  const goalname = ref('');
  const amount = ref('');
  const time = ref('');
  const linkaccount = ref('')
  const today = new Date().toLocaleDateString('en-CA')

  console.log("display", today)
  
  const isDisplay = ref(true);
  const emits = defineEmits(['close-modal']);
  
  function closeModal() {
    isDisplay.value = false;
    emits('close-modal');
  }
  
  async function addGoal() {
    // Convert string inputs to numbers using parseFloat
    const currentAmount = parseFloat(amount.value);
  
    // Create the request payload
    const requestData = {
        goal_name: goalname.value,
        connected_account_name: linkaccount.value,
        goal_value: currentAmount,
        goal_end_date: time.value,
        goal_created_date:today,
    };
  
    // Log the data before making the request
    console.log('Sending data to server:', requestData);
  
    // Make the HTTP request
    try {
      const response = await axios.post(`http://localhost:8080/api/v1/goal_saving_setting/${user_id}`, requestData, {
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
  onMounted(() => {
    addGoal();
  });
  </script>
  
  <style scoped>
    /* Your component styles go here */
  </style>