<script setup lang="ts">
import { Button } from '../../@/components/ui/Button'
import {
  Pagination,
  PaginationEllipsis,
  PaginationFirst,
  PaginationLast,
  PaginationList,
  PaginationListItem,
  PaginationNext,
  PaginationPrev,
} from '../../@/components/ui/pagination'


import { ref, onMounted, watch , watchEffect} from 'vue';
import axios from 'axios';

import { computed } from 'vue';
const investmentData = ref([]);
const currentPage = ref(1);
const itemsPerPage = 3; // Adjust the number of items per page as needed



const fetchData = async () => {
  try {
    const userId = localStorage.getItem('userId');
    const response = await axios.get(`http://localhost:8080/api/v1/bank_account/${userId}/Investment`, {
      withCredentials: true, // Include credentials in the request
    });

    investmentData.value = response.data;
    console.log('Investment data:', investmentData.value);
  } catch (error) {
    console.error('Error fetching investment accounts:', error);
  }
};

onMounted(fetchData);


// Watcher for updating data when currentPage changes
watch(() => currentPage.value, () => {
  fetchData();
});

// Computed property to calculate the total sum of current balances
const totalCurrentBalance = computed(() => {
  return investmentData.value.reduce((sum, account) => sum + account.current_balance, 0);
});

// Pagination
const paginate = (array, page_size, page_number) => {
  return array.slice((page_number - 1) * page_size, page_number * page_size);
};

const progress = ref(13)
watchEffect((cleanupFn) => {
  const timer = setTimeout(() => progress.value = 66, 500)
  cleanupFn(() => clearTimeout(timer))
})

</script>


<template>
     <header class="  h-16 border-t ps-5  flex items-center w-full ">
    
    <h1 class="font-semibold text-lg w-full flex items-center "><span class="w-max">Investment Balance</span>
        <Button class="w-6 h-6 p-0 ms-4 bg-indigo-700 align-middle">
         +
          </Button>
    </h1>
    <span class="justify-self-end text-indigo-700 py-1 px-4 rounded-md   font-bold text-right w-full"> Total: $ {{ totalCurrentBalance  }} </span>

  </header>
    <!-- component -->
    <div class="py-3 ps-8 shadow-lg shadow-slate-200 rounded-lg">
    <div v-for="account in paginate(investmentData, itemsPerPage, currentPage)" :key="account.id">
      <!-- Render each investment account -->
      <div class="flex items-center justify-between">
        <span class="text-sm text-slate-500">${{ account.current_balance }} - {{ account.account_name }}</span>
        <span class="px-2 py-1 bg-indigo-100 rounded-lg text-xs text-indigo-700 font-medium min-w-[46px] text-center">
          {{ account.bank_name }}
        </span>
      </div>
      <div class="w-full bg-slate-100 h-8 mb-6 mt-2 rounded-md">
        <div class="bg-indigo-700 h-8 rounded" :style="{ width: `${(account.current_balance / totalCurrentBalance) * 100}%` }"></div>
      </div>
    </div>


 
 

</div>

<div>
    <Pagination
        v-slot="{ page }"
        :total="Math.ceil(investmentData.length / itemsPerPage)*10"
        :sibling-count="1"
        :default-page="1"
        show-edges
        v-model="currentPage"
        
        class="flex items-center justify-center"

      >
        <PaginationList v-slot="{ items }" class="flex items-center gap-1">
          <PaginationFirst  @click="currentPage=1" class="w-6 h-6 bg-transparent" />
          <PaginationPrev @click="currentPage-=1" class="w-6 h-6 bg-transparent" />

          <template v-for="(item, index) in items">
            <PaginationListItem v-if="item.type === 'page'" :key="index" :value="item.value" as-child>
              <Button @click="currentPage=item.value" class="w-6 h-6 p-0 text-xs" :variant="item.value === page ? 'default' : 'outline'">
                {{ item.value }}
              </Button>
            </PaginationListItem>
            <PaginationEllipsis v-else :key="item.type" :index="index" />
          </template>

          <PaginationNext    @click="currentPage+=1" class="w-6 h-6 bg-transparent" />
          <PaginationLast @click="currentPage = Math.ceil(investmentData.length / itemsPerPage)" class="w-6 h-6 bg-transparent" />
        </PaginationList>
      </Pagination>
  </div>
</template>