<template>
    <div class="form">
    <fwb-modal v-if="isShowModal" @close="closeModal">
      <template #header>
        <div class="flex items-center text-lg p-4">
          Edit your profile
        </div>
      </template>
      <template #body>
        <div>
          <fwb-input
            v-model="user_address"
            placeholder="enter your address"
            label="Address"
            class="p-2"
          />
          <fwb-input
            v-model="num"
            placeholder="enter your phone number"
            label="Phonenum"
            class="p-2"
          />
          <div class="flex justify-between p-2">
            <fwb-button @click="closeModal" color="alternative">
              Cancel
            </fwb-button>
            <fwb-button @click="editPro" color="alternative">
              Save
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
let info= toRefs(data);
let pass = ref(info[0])
let num = ref(info[1])
let user_address = ref(info[2])


const isShowModal = ref(true);

const emits = defineEmits(['close-modal']);

function closeModal() {
  isShowModal.value = false;
  emits('close-modal');
}

async function editPro() {
  // Convert string inputs to numbers using parseFloat
  const phone = parseFloat(num.value);

  // Create the request payload
  const requestData = {
    number: phone,
    password: pass.value,
    address: user_address.value,
  };
  console.log('Sending data to server:', info);
  // Log the data before making the request
  console.log('Sending data to server:', requestData);

  // Make the HTTP request
  try {
    const response = await axios.put(`http://localhost:8080/api/v1/profile/${user_id}`, requestData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    withCredentials: true, // Include credentials in the request
    });

    console.log('check:', response.data);
    closeModal();

    // Handle success or update UI accordingly
  } catch (error) {
    console.error('Error:', error.response.data);

    // Handle error or update UI accordingly
  }
}
  onMounted(() => {
    editPro();
  });
</script>