<template>
  <div class="form">
    <fwb-modal v-if="isShowModal" @close="closeModal">
      <template #header>
        <div class="flex items-center text-lg p-4">
          Add your account
        </div>
      </template>
      <template #body>
        <div v-if='!showFileUploadSection'>
          <fwb-input
            v-model="youraccount"
            placeholder="enter your account"
            label="Account Name"
            class="p-2"
          />
          <fwb-select
            v-model="selected"
            :options="type"
            label="Select account type"
            class="p-2"
          />
          <fwb-input
            v-model="balance"
            placeholder="$1100"
            label="Balance"
            class="p-2"
          />
          <fwb-select
            v-model="accountbank"
            :options="bank"
            label="Bank"
            class="p-2"
          />
          <div @click="toggleSection"><h3>Or import CSV</h3></div> 
          <div class="flex justify-between p-2">
            <fwb-button @click="closeModal" color="alternative">
              Cancel
            </fwb-button>
            <fwb-button @click="createAccount" color="alternative">
              Save
            </fwb-button>
          </div>
        </div>
        <div v-if="showFileUploadSection" class="items-center justify-center w-full p-4">
          <label for="dropzone-file" class="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600">
            <div class="flex flex-col items-center justify-center pt-5 pb-6">
              <svg class="w-5 h-5 mb-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
              </svg>
              <p class="mb-2 text-sm text-gray-500 dark:text-gray-400"><span class="font-semibold">Click to upload</span> or drag and drop</p>
              <p class="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG or GIF (MAX. 800x400px)</p>
            </div>
            <input id="dropzone-file" type="file" class="hidden" @change="handleFileChange"/>
          </label>
          <div class="pt-5" @click="toggleSection"><h3>Add account Manually</h3></div>
          <div class="flex justify-between p-2">
            <fwb-button @click="closeModal" color="alternative">
              Cancel
            </fwb-button>
            <fwb-button @click="importFile" color="alternative">
              Import
            </fwb-button>
          </div>
        </div> 
      </template>
    </fwb-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, defineEmits } from 'vue';
import { FwbButton, FwbModal, FwbInput, FwbSelect } from 'flowbite-vue';
import axios from 'axios';

const user_id = '657deedb53a90ee98e224654';
const youraccount = ref('');
const balance = ref('');
const accountbank = ref('');
const showFileUploadSection = ref(false);
const selected = ref('');
const type = [
  { value: 'Expense', name: 'Expense' },
  { value: 'Saving', name: 'Saving' },
  { value: 'Investment', name: 'Investment' },
];
const bank = [
  { value: 'VIB', name: 'VIB' },
  { value: 'VIETCOMBANK', name: 'VIETCOMBANK' },
  { value: 'TECHCOMBANK', name: 'TECHCOMBANK' },
]

const isShowModal = ref(true);
let selectedFile = null;

const emits = defineEmits(['close-modal']);

function closeModal() {
  isShowModal.value = false;
  emits('close-modal');
}

async function createAccount() {
  // Convert string inputs to numbers using parseFloat
  const currentBalance = parseFloat(balance.value);

  // Create the request payload
  const requestData = {
    account_name: youraccount.value,
    account_type: selected.value,
    current_balance: currentBalance,
    bank_name: accountbank.value,
  };

  // Log the data before making the request
  console.log('Sending data to server:', requestData);

  // Make the HTTP request
  try {
    const response = await axios.post(`http://localhost:8080/api/v1/bank_account/${user_id}/create_manually`, requestData, {
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

async function importFile() {
  // Check if a file is selected
  if (!selectedFile) {
    console.error('No file selected for import');
    return;
  }

  const formData = new FormData();
  formData.append('csv_file', selectedFile);

  try {
    const response = await axios.post(`http://localhost:8080/api/v1/bank_account/${user_id}/bank_account_import`, formData, {
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

function handleFileChange(event) {
  selectedFile = event.target.files[0];
}

function toggleSection() {
  showFileUploadSection.value = !showFileUploadSection.value;
}
</script>

<style scoped>
  /* Your component styles go here */
</style>
