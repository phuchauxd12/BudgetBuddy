<template>
    <Links/>
    <AddIncome class="form" v-if="isVisible2" @close-modal="closeForm2" :newmonth="newmonth"/>
    <AddSub class="form" v-if="isVisible3" @close-modal="closeForm3" :newmonth="newmonth"/>
    <AddPlan class="form" v-if="isVisible4" @close-modal="closeForm4" :newmonth="newmonth"/>
    <AddGoal class="form" v-if="isVisible5" @close-modal="closeForm5"/>
    <AddBill class="form" v-if="isVisible8" @close-modal="closeForm8"/>
    <EditIncome class="form" v-if="isVisible6" @close-modal="closeForm6" :data="newdata"/>
    <EditSub class="form" v-if="isVisible7" @close-modal="closeForm7" :data="newdata"/>
    <EditBill class="form" v-if="isVisible9" @close-modal="closeForm9" :data="newdata"/>
    <EditGoal class="form" v-if="isVisible10" @close-modal="closeForm10" :data="newdata"/>

    <section class="planning">
      <div class="networth">
        <div class="total">
          <p class='label'>Planning for</p>
        </div>
        <div class="overview-section bg-violet-600" @click="showComponent('incomeAfterBill')" >
          <h2 class="text-xl text-white font-bold">{{ inc - bil - sub }}</h2>
          <p class="text-xs text-white">Income after bill</p>
        </div>
        <div class="overview-section bg-neutral-100" @click="showComponent('plannedSpending')">
          <h2 class="text-xl text-black font-bold">{{ plannedSpending }}</h2>
          <p class="text-xs text-black">Planned Spending</p>
        </div>
        <div class="overview-section bg-blue-600" @click="showComponent('savingGoal')">
          <h2 class="text-xl text-white font-bold">{{ savingGoals }}</h2>
          <p class="text-xs text-white">Saving Goals</p>
        </div>
      </div>
      <div class="screen">
        <!-- <Test v-if="selectedComponent === 'incomeAfterBill'" @edit="collect($event)"/> -->
        <IncomeafterBill v-if="selectedComponent === 'incomeAfterBill'" @add-income="receiveEmit2($event)" @add-sub="receiveEmit3($event)" @add-bill="receiveEmit8" @edit-income="collect1($event)" @edit-sub="collect2($event)" @edit-bill="collect3($event)" @send-sub="rec2($event)" @send-bil="rec3($event)" @send-in="rec1($event)" />
        <Plan2 v-else-if="selectedComponent === 'plannedSpending'" @add-plan="receiveEmit4($event)"  @view-plan="handleView($event)" @send-total-1="receiveTotal1($event)"/>
        <SavingGoals v-else-if="selectedComponent === 'savingGoal'" @add-goal="receiveEmit5" @edit-goal="collect4($event)" @send-total-goal="receiveTotal2($event)"/>
        <SpecificPlan v-else-if="selectedComponent === 'viewSpecificPlan'" :data="newdata" @go-back="showComponent('plannedSpending')"/>
      </div>
    </section> 
</template>

<script setup lang="ts">
  import { ref, Ref, computed, watch } from 'vue'
  import IncomeafterBill from '../components/IncomeafterBill.vue'
  import Plan2 from '../components/PlannedSpending.vue'
  import SavingGoals from '../components/SavingGoals.vue'
  import AddBill from '../components/AddBill.vue'
  import EditBill from '../components/EditBill.vue'
  import AddIncome from '../components/AddIncome.vue'
  import AddSub from "../components/AddSubcriptions.vue" 
  import AddPlan from "../components/AddMonthlySpendingPlan.vue"
  import AddGoal from "../components/AddGoal.vue"
  import EditIncome from "../components/EditIncome.vue"
  import EditSub from '../components/EditSub.vue'
  import SpecificPlan from '../components/SpecificPlan.vue'
  import EditGoal from '../components/EditGoal.vue'
  import Links from  '../components/Link.vue';
  let plannedSpending = ref(null);
  let savingGoals = ref(null);
  let selectedComponent: Ref<string> = ref('incomeAfterBill');

  const isVisible2 = ref(false);
  const isVisible3 = ref(false);
  const isVisible4 = ref(false);
  const isVisible5 = ref(false);
  const isVisible6 = ref(false);
  const isVisible7 = ref(false);
  const isVisible8 = ref(false);
  const isVisible9 = ref(false);
  const isVisible10 = ref(false);
  let newdata = ref(null)
  let newmonth = ref(null)
  let inc = ref(null)
  let sub = ref(null)
  let bil = ref(null)
  let total = ref(null);


  const handleView = (event) => {
    newdata = event
    showComponent('viewSpecificPlan');
  }
  const rec1 = (event) => {
    inc = event
    console.log('see income', inc)
  }
  const rec3 = (event) => {
    bil = event
    console.log('see bil', bil)
  }
  const rec2 = (event) => {
    sub = event
    console.log('see sub', sub)
  }
  const collect1 = (event) => {
    newdata = event
    console.log("see1", newdata)
    isVisible6.value = true;
  }
  const collect2 = (event) => {
    newdata = event
    console.log("see2", newdata)
    isVisible7.value = true;
  }
  const collect3 = (event) => {
    newdata = event
    console.log("see3", newdata)
    isVisible9.value = true;
  }
  const collect4 = (event) => {
    newdata = event
    console.log("see4", newdata)
    isVisible10.value = true;
  }
  const receiveEmit2 = (event) => {
    newmonth = event
    console.log("seemonth", newmonth)
    isVisible2.value = true;
  }
  const receiveEmit3 = (event) => {
    newmonth = event
    console.log("seemonthinsub", newmonth)
    isVisible3.value = true;
  }
  const receiveEmit4 = (event) => {
    newmonth = event
    console.log("seemonthinplan", newmonth)
    isVisible4.value = true;
  }
  const receiveEmit5 = () => {
    isVisible5.value = true;
  }
  const receiveEmit8 = () => {
    isVisible8.value = true;
  }
  const receiveTotal1 = (event) => {
    plannedSpending = event
    console.log("amount", plannedSpending)
  }
  const receiveTotal2 = (event) => {
    savingGoals = event
    console.log("amount", savingGoals)
  }
  const closeForm2 = () => {
    isVisible2.value = false;
  }
  const closeForm3 = () => {
    isVisible3.value = false;
  }
  const closeForm4 = () => {
    isVisible4.value = false;
  }
  const closeForm5 = () => {
    isVisible5.value = false;
  }
  const closeForm6 = () => {
    isVisible6.value = false;
  }
  const closeForm7 = () => {
    isVisible7.value = false;
  }
  const closeForm8 = () => {
    isVisible8.value = false;
  }
  const closeForm9 = () => {
    isVisible9.value = false;
  }
  const closeForm10 = () => {
    isVisible10.value = false;
  }
  const showComponent = (componentName: string) => {
    selectedComponent.value = componentName;
  };
</script>

<style scoped>
  
.overview-section {
  cursor: pointer;
  padding: 10px;
  border-radius: 20px;
  margin: 10px;
}
.form{
  position: relative;
  z-index: 999;
}

</style>