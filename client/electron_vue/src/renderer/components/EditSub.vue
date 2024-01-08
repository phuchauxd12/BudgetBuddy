<template>
    <div class="form">
      <fwb-modal v-if="isDisplay" @close="closeModal">
        <template #header>
          <div class="flex items-center text-lg p-4">
            Edit Subscription
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
              v-model="Samount"
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
  import { FwbButton, FwbModal, FwbInput, FwbSelect } from 'flowbite-vue';
  import axios from 'axios';
  
  const user_id = '657deedb53a90ee98e224654';
  let { data } = defineProps(['data']);
  let { id, spending_name, amount, time_duration } = toRefs(data);
  let spending_id = ref(id.value);
  let subname = ref(spending_name.value);
  let Samount = ref(amount.value);
  let time = ref(time_duration.value);
  const type = ref('Subscription')
  
  const isDisplay = ref(true);
  const emits = defineEmits(['close-modal']);
  
  function closeModal() {
    isDisplay.value = false;
    emits('close-modal');
  }

  async function addSub() {
    // Convert string inputs to numbers using parseFloat
    const currentAmount = parseFloat(Samount.value);

    if (!data) {
    console.error('Data prop not available');
    return;
    }
  
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
      const response = await axios.put(`http://localhost:8080/api/v1/plan_spending/${user_id}/${spending_id.value.toString()}`, requestData, {
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
    addSub();
    console.log('test props:', data);
  });
  </script>
  
  <style scoped>
    /* Your component styles go here */
  </style>