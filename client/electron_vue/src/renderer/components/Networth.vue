
<template>
  <section class="networth">
    <div class="total">
      <p class="label">Networth</p>
      <h2 class="label">$0</h2>
    </div>
      <div class="networth_container">
        <article class="bank_acc">
          <button @click="isShow = !isShow">
            <p>Bank</p>
            <p>$0</p>
          </button>
          <div v-if="isShow">Checking 1</div>
        </article>
        <article class="bank_acc">
          <button @click="isShow2 = !isShow2">
            <p>Cash</p>
            <p>$0</p>
          </button>
          <div v-if="isShow2">Checking 1</div>
        </article>
        <article class="bank_acc">
          <button @click="isShow3 = !isShow3">
            <p>Credit</p>
            <p>$0</p>
          </button>
          <div v-if="isShow3">Checking 1</div>
        </article>
        <article class="bank_acc">
          <button @click="isShow4 = !isShow4">
            <p>Saving</p>
            <p>$0</p>
          </button>
          <div v-if="isShow4">Checking 1</div>
        </article>
        <article class="bank_acc">
          <button @click="isShow5 = !isShow5">
            <p>Investment</p>
            <p>$0</p>
          </button>
          <div v-if="isShow5">Checking 1</div>
        </article>

        <button @click="addAccount" class="add">Add</button>
      </div>
  </section>
</template>
<script>
  import Swal from 'sweetalert2';

  export default {
    name: "ToggleDiv",
    data: function () {
      return {
        isShow: false,
        isShow2: false,
        isShow3: false,
        isShow4: false,
        isShow5: false,
      };
    },

    props: {
      msg: String,
    },

    methods:{
      addAccount(){
        Swal.fire({
          title: 'Error!',
          text: 'Do you love it',
          icon: 'error',
          confirmButtonText: 'Cool'
        });
      }
    }

  };
</script>
<style scoped>
  .notShown {
    visibility: hidden !important;
  }
</style>


