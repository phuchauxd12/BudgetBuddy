<script setup lang="ts">
import axios from 'axios';
import { ref, onMounted, watch } from 'vue';

const stocks = ref([]);
const originalStocks = ref([]);
const activeButton = ref('hot');

const setActiveButton = (button) => {
  stocks.value = originalStocks.value;
  activeButton.value = button;
  console.log(activeButton.value);
};

const fetchStockData = async () => {
  try {
    const response = await axios.get('http://localhost:8080/api/v1/get_all_stocks');
    stocks.value = response.data.list_of_stocks;
    originalStocks.value = response.data.list_of_stocks;
    console.log(stocks.value);
    
  } catch (error) {
    console.error('Error fetching stock data:', error);
  }
};

onMounted(fetchStockData);

const filterUpTrendStock = () => {
  setActiveButton('upside');
 
  stocks.value = originalStocks.value.filter(stock => stock.price_diff > 0);
 
};

const filterDownTrendStock = () => {
  setActiveButton('downside');

  stocks.value = originalStocks.value.filter(stock => stock.price_diff < 0);
};
</script>

<template>
    <header class="flex-none flex h-16 border-t px-4 items-center">
    
    <h1 class="font-semibold text-lg">Market Status</h1>

  </header>
<header class="flex-none flex px-4 items-center">

   <a href="#" :class="{ 'inline-block rounded-full text-indigo-700 bg-gray-100 text-xs font-bold mr-1 md:mr-2 mb-2 px-2 md:px-4 py-1': activeButton === 'hot', 'inline-block rounded-full text-black text-xs mr-1 md:mr-2 mb-2 px-2 md:px-4 py-1': activeButton !== 'hot' }" @click="setActiveButton('hot')" >
                         Hot
                      </a>
       <a href="#" :class="{ 'inline-block rounded-full text-indigo-700 bg-gray-100 text-xs font-bold mr-1 md:mr-2 mb-2 px-2 md:px-4 py-1': activeButton === 'upside', 'inline-block rounded-full text-black text-xs mr-1 md:mr-2 mb-2 px-2 md:px-4 py-1': activeButton !== 'upside' }"
       @click="filterUpTrendStock()
                          ">
                          Upside
                      </a>
       <a href="#"  :class="{ 'inline-block rounded-full text-indigo-700 bg-gray-100 text-xs font-bold mr-1 md:mr-2 mb-2 px-2 md:px-4 py-1': activeButton === 'downside', 'inline-block rounded-full text-black text-xs mr-1 md:mr-2 mb-2 px-2 md:px-4 py-1': activeButton !== 'downside' }" @click="filterDownTrendStock()">
                          Downside
                      </a>
  </header>

  <ul class="flex flex-col p-4 overflow-hidden h-72 overflow-y-scroll">
    <li v-for="stock in stocks" :key="stock.stock_symbol" class="border-gray-400 flex flex-row mb-2">
  <div class="select-none cursor-pointer bg-gray-50 rounded-md flex flex-1 items-center p-3 transition duration-200 ease-in-out transform hover:-translate-y-1 hover:shadow">
    <div class="flex flex-col pl-1 mr-16">
      <div class="font-medium text-sm text-black w-max">{{ stock.stock_name }}</div>
      <div class="text-gray-600 text-xs justify-self-start text-left">
        <a href="#" class="inline-block rounded-full text-white bg-yellow-700 text-xs font-bold mr-1 md:mr-2 mb-2 px-2 md:px-4 py-1">
          {{ stock.stock_symbol }}
        </a>
      </div>
    </div>
    <div class="flex flex-col justify-items-end w-full">
      <div class="text-black text-sm font-medium w-full text-right">
        $ {{ stock.current_price }}</div>
      <div class="text-blue-500 text-sm font-medium w-full text-right" v-if="stock.previous_close < stock.current_price">
        ↑ {{ stock.price_diff }}%
      </div>
     
      <div class="text-red-500 text-sm font-medium w-full text-right" v-else>
        ↓ {{ stock.price_diff }}%
      </div>
    </div>
  </div>
</li>

  </ul>
</template>