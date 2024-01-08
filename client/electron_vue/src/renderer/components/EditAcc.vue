<template>
    <div class="form">
    <fwb-modal v-if="isShowModal" @close="closeModal">
      <template #header>
        <div class="flex items-center text-lg p-4">
          Add your account
        </div>
      </template>
      <template #body>
        <div>
          <fwb-input
            v-model="youraccount"
            placeholder="enter your account"
            label="Account Name"
            class="p-2"
          />
          <fwb-select
            v-model="accountbank"
            :options="bank"
            label="Bank"
            class="p-2"
          />
          <div class="flex justify-between p-2">
            <fwb-button @click="closeModal" color="alternative">
              Cancel
            </fwb-button>
            <fwb-button @click="editAcc" color="alternative">
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
let ac= toRefs(data);
let youraccount = ref(ac[0])
let balance = ref(ac[1])
let accountbank = ref(ac[3])
let selected = ref(ac[4])
const bank = [
  { value: 'VIB', name: 'VIB' },
  { value: 'VIETCOMBANK', name: 'VIETCOMBANK' },
  { value: 'TECHCOMBANK', name: 'TECHCOMBANK' },
]

const isShowModal = ref(true);

const emits = defineEmits(['close-modal']);

function closeModal() {
  isShowModal.value = false;
  emits('close-modal');
}

async function editAcc() {

  const requestData = {
    new_account_name: youraccount.value,
    new_bank_name: accountbank.value,
  };

  try {
    const response = await axios.put(`http://localhost:8080/api/v1/${user_id}/${selected.value}/${youraccount.value}`, requestData, {
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
    editAcc();
  });
</script>