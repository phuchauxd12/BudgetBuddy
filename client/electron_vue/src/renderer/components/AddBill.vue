<template>
    <div class="form">
      <fwb-modal v-if="isDisplay" @close="closeModal">
        <template #header>
          <div class="flex items-center text-lg p-4">
            Add Bill
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
  import { FwbButton, FwbModal, FwbInput, FwbRadio } from 'flowbite-vue';
  import axios from 'axios';
  
  const user_id = '657deedb53a90ee98e224654';
  const billname = ref('');
  const amount = ref('');
  const time = ref('');
  const picked = ref('Yes')
  
  const isDisplay = ref(true);
  const emits = defineEmits(['close-modal']);
  
  function closeModal() {
    isDisplay.value = false;
    emits('close-modal');
  }
  
  async function addBill() {
    const billAmount = parseFloat(amount.value);
  
    const requestData = {
      bill_name: billname.value,
      bill_value: billAmount,
      recurrent_reminder: picked.value,
      recurrent_date_value: time.value,

    };
  
    console.log('Sending data to server:', requestData);
  
    try {
      const response = await axios.post(`http://localhost:8080/api/v1/user_bill/${user_id}/create`, requestData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      withCredentials: true, 
      });
  
      console.log('Server response:', response.data);
      
      closeModal();

    } catch (error) {
      console.error('Error:', error.response.data);
  
    }
  }
  onMounted(() => {
    addBill();
  });
  </script>
  
  <style scoped>
    /* Your component styles go here */
  </style>