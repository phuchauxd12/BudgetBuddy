<script setup lang="ts">



import { Button } from '../../@/components/ui/Button'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '../../@/components/ui/select'
import {   ref, watch,  computed, reactive, nextTick, onMounted} from 'vue'
import axios from 'axios'
import { Bar, Line } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, PointElement, LineElement, CategoryScale, LinearScale } from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale,  PointElement,
  LineElement)


const updateTrigger = reactive({
  forceUpdate: 0,
});

const methodThatForcesUpdate = () => {
  updateTrigger.forceUpdate += 1;
 
};

const randomColor = () => {
  // Generate a random hex color code
  return '#' + Math.floor(Math.random()*16777215).toString(16);
};

const userName = ref<string | null>(null);
const year = ref(2023);
const selectedMonth = ref('01');
const user_id = localStorage.getItem('userId');
const transactions = ref([]);
const previousTransactions = ref([]);
const percentageChange = ref(0);
const finalBudget = ref(0);
const finalBudgetPerDay = ref([]);
const forecastData = ref([]);



const activeButton = ref('forecast');

const toggleButton = (button: string) => {
  console.log(button);
  activeButton.value = activeButton.value !== button ? button : null;
};



const fetchData = async (month: string | undefined, year: string | undefined) => {
  try {
   
    console.log(year, month)
    const response = await axios.get(`http://localhost:8080/api/v1/transaction/${user_id}/${year}-${month}`, {
        withCredentials: true,
        
    });
    // Handle the response data here
    transactions.value = response.data;
    console.log('Report Component:', transactions.value);
  } catch (error) {
    // Handle the error here
    console.error(error);
     transactions.value = [];
    finalBudget.value = 0;
  }
};

const fetchPreviousData = async (prevMonth: string | undefined, prevYear: string | undefined) => {
  try {
    const response = await axios.get(`http://localhost:8080/api/v1/transaction/${user_id}/${prevYear}-${prevMonth}`, {
        withCredentials: true,
    });
    previousTransactions.value = response.data;
    console.log('Previous Transactions:', previousTransactions.value);

    // Calculate percentage change
    calculatePercentageChange();
  } catch (error) {
    console.error(error);
    
    percentageChange.value = 0;
  }
};



const calculateFinalBudget = (transactionData) => {
  let budget = 0;

  transactionData.forEach((transaction) => {
    if (transaction.transaction_type === 'Income') {
        budget += transaction.Amount;
    } else if (transaction.transaction_type === 'Outcome') {
        budget -= transaction.Amount;
    }
  });
return budget;
  
  
};

const calculateFinalBudgetPerDay = (transactionData) => {
  const month  = parseInt(selectedMonth.value) - 1;
  console.log('Month afeter -1',month)
  const daysInMonth = new Date(year.value, month, 0).getDate();
  const dailyBudget = Array.from({ length: daysInMonth }, (_, i) => {
    const day = (i + 1).toString().padStart(2, '0');
    const dailyTransactions = transactionData.filter(transaction => transaction.transaction_date.endsWith(`-${day}`));
    return calculateFinalBudget(dailyTransactions);
  });
  return dailyBudget;
};


const calculatePercentageChange = () => {
   console.log(finalBudget.value);
  const previousBudget = calculateFinalBudget(previousTransactions.value);
  const currentBudget = finalBudget.value;
  console.log(currentBudget, previousBudget);
  // Calculate percentage change
  percentageChange.value = Math.ceil(((currentBudget - previousBudget) / Math.abs(previousBudget)) * 100);
};

watch(transactions, (newTransactions) => {
    
    // render chart again
    finalBudgetPerDay.value = calculateFinalBudgetPerDay(newTransactions);
console.log('Watch final daily budget', finalBudgetPerDay.value);
    chartData.value.labels =  generateDateLabels(year.value, selectedMonth.value);
    chartData.value.datasets[0].data = finalBudgetPerDay.value;
    console.log('Chart datasets reload:', chartData.value);   


    // render stats again
  finalBudget.value = calculateFinalBudget(newTransactions);
  const prevMonth = String(Number(selectedMonth.value) - 1).padStart(2, '0');
  const prevYear = String(year.value);
  fetchPreviousData(prevMonth, prevYear);
  chartData.value.datasets[0].backgroundColor = finalBudgetPerDay.value.map(() => randomColor()) as any[];

  methodThatForcesUpdate()
 
});
watch([selectedMonth, year], ([newSelectedMonth, newYear]) => {
    chartData.value.labels = generateDateLabels(newYear, newSelectedMonth);
    console.log('Watch', chartData.value.labels);
  fetchData(newSelectedMonth, newYear.toString());
  
});

const selectedMonthName = computed(() => {
  const monthNames = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
  ];

  const monthIndex = parseInt(selectedMonth.value) - 1;
  return monthNames[monthIndex] || '';
});

const generateDateLabels = (year, month) => {
    console.log('GenerateDateLabels',year, month)
  const daysInMonth = new Date(year, month, 0).getDate();
  const labels = Array.from({ length: daysInMonth }, (_, i) => {
    const day = (i + 1).toString().padStart(2, '0');
    return `${day}`;
  });
 
  return labels;
};
const chartData = ref({
  labels: generateDateLabels(year.value, selectedMonth.value),
  datasets: [
    {
      data: [],
      backgroundColor: [],
       // Array to hold random colors for each bar
    },
  ],
});

const fetchForecastData = async () => {
  try {


    // Fetch forecast data
    const forecastResponse = await axios.get(`http://localhost:8080/api/v1/prediction/${user_id}`, {
      withCredentials: true,
    });
    // Handle the forecast data here
    forecastData.value = forecastResponse.data;
    console.log('Forecast Data:', forecastData.value);
    forecastChartData.value.datasets[0].data = forecastResponse.data;
    console.log(forecastChartData.value)
    methodThatForcesUpdate()
    return forecastData.value;
  } catch (error) {
    // ... existing code ...
  }
};



const chartOptions = ref({
  responsive: true,
  plugins: {
    title: {
      display: true,
      text: 'Monthly Overall Chart',
      font: {
        size: 14,
      },
    },
  },
  scales: {
    x: {
      title: {
        display: true,
        text: 'Days in a Month',
        font: {
          size: 12,
        },
      },
    },
    y: {
      title: {
        display: true,
        text: 'Final Budget Per Day',
        font: {
          size: 12,
        },
      },
    },
  },
 // Set to false to allow custom width and height
 // Set your desired height
});

const forecastChartData = ref({
  labels: Array.from({ length: 94 }, (_, i) => (i + 1).toString()),
  datasets: [
    {
      data: fetchForecastData,
      borderColor: 'rgba(49, 156, 87, 1)', // Adjust color as needed
      borderWidth: 2,
    },
  ],
});


const forecastChartOptions = ref({
  responsive: true,
  plugins: {
    title: {
      display: true,
      text: 'Forecast Overall Chart',
      font: {
        size: 14,
      },
    },
  },
  scales: {
    x: {
      title: {
        display: true,
        text: 'Days in a Month',
        font: {
          size: 12,
        },
      },
    },
    y: {
      title: {
        display: true,
        text: 'Final Budget Per Day',
        font: {
          size: 12,
        },
      },
    },
  },
 // Set to false to allow custom width and height
 // Set your desired height
});
onMounted(fetchForecastData);
</script>

<template>
   
  <section class="display">

    <div class="grid grid-cols-5 gap-4 h-full">
        
  <div class="chart col-span-3 ps-8 pt-10 pe-4">
   
    <div class="time">
        
        <h1 class="year-picker font-semibold text-2xl w-full flex items-center pb-4 "><Button @click="year= year - 1" class="w-6 h-6 p-0 me-4 bg-indigo-700 align-middle text-white">
         -
          </Button><span class="w-max">{{year}}</span>
        <Button  @click="year= year + 1" class="w-6 h-6 p-0 ms-4 bg-indigo-700 align-middle text-white">
         +
          </Button>
    </h1>
   
        <Select class="month-picker" v-model="selectedMonth" >
    <SelectTrigger>
      <SelectValue placeholder="Select a month" />
    </SelectTrigger>
    <SelectContent>
      <SelectGroup>
        <SelectLabel>{{year}}</SelectLabel>
        <SelectItem value="01">
          January
        </SelectItem>
        <SelectItem value="02">
          February
        </SelectItem>
        <SelectItem value="03">
          March</SelectItem>
        <SelectItem value="04">April</SelectItem>  
        <SelectItem value="05">May</SelectItem>
        <SelectItem value="06">June</SelectItem>
        <SelectItem value="07">July</SelectItem>
        <SelectItem value="08">August</SelectItem>
        <SelectItem value="09">September</SelectItem>
        <SelectItem value="10">October</SelectItem>
        <SelectItem value="11">November</SelectItem>
        <SelectItem value="12">December</SelectItem>
      </SelectGroup>
     
    </SelectContent>
  </Select>
  <div class="desc w-full text-left pt-4 "><p class="text-black text-xl font-semibold"><span class="text-indigo-800">{{ selectedMonthName }}</span> cost against budget</p></div>
    </div>
        <div class="bg-white text-sm text-gray-950 leading-none border-2 border-gray-200 rounded-full inline-flex mt-4 w-full  ">
    <button @click="toggleButton('budget')" :class="{ 'active': activeButton === 'budget' }" class="inline-flex items-center transition-colors duration-300 ease-in focus:outline-none hover:text-indigo-800 focus:text-indigo-800 rounded-l-full px-4 py-2 w-1/2" id="grid">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="fill-current w-4 h-4 mr-2">
        <rect x="3" y="3" width="7" height="7"></rect>
        <rect x="14" y="3" width="7" height="7"></rect>
        <rect x="14" y="14" width="7" height="7"></rect>
        <rect x="3" y="14" width="7" height="7"></rect>
      </svg>
      <span>Budget</span>
    </button>
    <button @click="toggleButton('forecast')" :class="{ 'active': activeButton === 'forecast' }" class="inline-flex items-center transition-colors duration-300 ease-in focus:outline-none hover:text-black focus:text-indigo-50 rounded-r-full px-4 py-2 w-1/2" id="list">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="fill-current w-4 h-4 mr-2">
        <line x1="8" y1="6" x2="21" y2="6"></line>
        <line x1="8" y1="12" x2="21" y2="12"></line>
        <line x1="8" y1="18" x2="21" y2="18"></line>
        <line x1="3" y1="6" x2="3.01" y2="6"></line>
        <line x1="3" y1="12" x2="3.01" y2="12"></line>
        <line x1="3" y1="18" x2="3.01" y2="18"></line>
      </svg>
      <span>Forecast</span>
    </button>
  </div>
<div class="pt-4 flex items-center justify-center ">
    <Bar
    v-if="activeButton === 'budget'"
    :key="updateTrigger.forceUpdate"
    id="my-chart-id"
    :options="chartOptions"
    :data="chartData"  
    class="w-full h-full"
    width="50px" height="40px"
  >Chart could not be load</Bar>
  <Line
      v-if="activeButton === 'forecast'"
      :key="updateTrigger.forceUpdate"
      id="forecast-chart-id"
      :options="forecastChartOptions"
      :data="forecastChartData"
      class="w-full h-full"
      width="50px"
      height="40px"
    >Forecast Chart could not be load</Line>
  </div>
  </div>
  <div class="overall-badge pe-8 py-5 pt-10 col-span-2 h-full ">
    <div class="stats bg-lime-950 w-full text-left">
  
        <div class="stat">
    <div class="stat-title">Compare to last month</div>
    <div class="stat-value">${{ finalBudget }}</div>
    <div v-if="percentageChange>=0" class="stat-desc">{{percentageChange}}% more than last month</div>
    <div v-else-if="percentageChange<0" class="stat-desc">{{percentageChange}}% less than last month</div>
  </div>
  
</div>
<div class="transaction h-96">
    <div class="h-full">
    <header class="h-16 border-t ps-5 flex items-center w-full">
      <h1 class="font-semibold text-lg w-full flex items-center h-max">
        <span class="w-max "> Transactions</span>
        <div class="w-full align-self-end">
          <Button class="h-6 bg-indigo-700 align-middle w-max ms-5 text-white text-left">
            Details
          </Button>
        </div>
      </h1>
    </header>
    <div class="relative flex flex-col h-full min-w-0 w-full mb-6 break-words bg-white border-0 shadow-md dark:bg-slate-850 dark:shadow-dark-xl rounded-2xl bg-clip-border overflow-x-scroll">
      <div class="p-6 px-4 pb-0 mb-0 border-b-0 rounded-t-2x"></div>
      <div class="flex-auto px-4 pb-4 overflow-y-scroll ">
        <ul class="flex flex-col pl-0 mb-0 rounded-lg">
          <li
            v-for="(transaction, index) in transactions"
            :key="index"
            class="relative flex justify-between py-2 pl-0 mb-2 border-0 rounded-t-inherit text-inherit rounded-xl"
          >
            <div class="flex items-center">
              <button
                v-if="transaction.transaction_type === 'Outcome'"
                class="leading-pro ease-in text-xs bg-150 w-5 h-5 p-1.2 rounded-full bg-x-25 mr-4 mb-0 flex cursor-pointer items-center justify-center border border-solid border-red-600 bg-transparent text-center align-middle font-bold uppercase text-red-600 transition-all hover:opacity-75"
              >
                ↓
              </button>
              <button
                v-else-if="transaction.transaction_type === 'Income'  || transaction.transaction_type === 'Investment'"
                class="leading-pro ease-in text-xs bg-150 w-5 h-5 p-1.2 rounded-full bg-x-25 mr-4 mb-0 flex cursor-pointer items-center justify-center border border-solid border-lime-600 bg-transparent text-center align-middle font-bold uppercase text-lime-600 transition-all hover:opacity-75"
              >
                ↑
              </button>
              <div class="flex flex-col">
                <h6 class="mb-1 text-xs leading-normal dark:text-white text-slate-700 font-bold text-left">
                  {{ transaction.transaction_name }} - {{ transaction.Payee }}
                </h6>
                <span class="text-xs leading-tight dark:text-white/80 font-light text-left">
                  {{ transaction.transaction_date }}
                </span>
              </div>
            </div>
            <div class="flex flex-col items-center justify-center">
              <p
                v-if="transaction.transaction_type === 'Outcome'"
                class="relative z-10 inline-block m-0 text-sm font-semibold leading-normal text-transparent bg-gradient-to-tl from-red-600 to-orange-600 bg-clip-text"
              >
                - ${{ transaction.Amount }}
              </p>
              <p
                v-else-if="transaction.transaction_type === 'Income'  || transaction.transaction_type === 'Investment'"
                class="relative z-10 inline-block m-0 text-sm font-semibold leading-normal text-transparent bg-gradient-to-tl from-green-600 to-lime-400 bg-clip-text"
              >
                + ${{ transaction.Amount }}
              </p>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </div>
</div>
  </div>
  

 
  
</div>
    
   
   
  

  </section>  
</template>

<style>
.display {
   
    width: 900px;
    height: 650px ;
    max-height: 100% ;

}
.active {
  background-color: rgb(57, 62, 202);
  border-radius: 9999px;
  color: white;
}

</style>