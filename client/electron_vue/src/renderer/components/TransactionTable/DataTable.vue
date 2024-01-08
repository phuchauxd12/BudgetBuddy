<script setup lang="ts" generic="TData, TValue">
import type { ColumnDef,SortingState, ColumnFiltersState } from '@tanstack/vue-table'
import {h, ref} from 'vue'
import {
    FlexRender,
    getCoreRowModel,
    useVueTable,
    getPaginationRowModel,
    getSortedRowModel,
    getFilteredRowModel,
    PaginationState
} from "@tanstack/vue-table"
import { Button } from "../../../@/components/ui/button"
import { valueUpdater } from '../../../@/lib/utils'
import { Input } from '../../../@/components/ui/input'

import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,

} from "../../../@/components/ui/table"

const props = defineProps<{
    columns: ColumnDef<TData, TValue>[]
    data: TData[]
}>()

const sorting = ref<SortingState>([])
    const columnFilters = ref<ColumnFiltersState>([])

const table = useVueTable({
    get data() { return props.data },
    get columns() { return props.columns },
    getCoreRowModel: getCoreRowModel(),
    
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: updaterOrValue => valueUpdater(updaterOrValue, sorting),
    onColumnFiltersChange: updaterOrValue => valueUpdater(updaterOrValue, columnFilters),
    getFilteredRowModel: getFilteredRowModel(),
    state: {
        get sorting() { return sorting.value },
        get columnFilters() { return columnFilters.value },
        
    },
    initialState: {
        pagination: {
            pageSize: 3,
        },
    },
})
</script>

<template>
    <div>
        <div class=" flex flex-row w-full text-right items-center justify-between">
            <Input class="max-w-sm " placeholder="Filter categories..."
                :model-value="table.getColumn('categories')?.getFilterValue() as string"
                @update:model-value=" table.getColumn('categories')?.setFilterValue($event)" />
                 <div class="flex items-center justify-end py-4 space-x-2">
      <Button
          variant="outline"
          class="bg-indigo-800 text-white w-max"
          size="sm"
          :disabled="!table.getCanPreviousPage()"
          @click="table.previousPage()"
        >
          Previous
        </Button>
        <Button
          variant="outline"
          class="text-indigo-800 bg-white w-max"
          size="sm"
          :disabled="!table.getCanNextPage()"
          @click="table.nextPage()"
        >
          Next
        </Button>
      </div>
        </div>
    <div class="border rounded-md">
        <Table>
            <TableHeader class="bg-indigo-700 text-ingdigo-950 ">
                <TableRow  v-for="headerGroup in table.getHeaderGroups()" :key="headerGroup.id">
                    <TableHead class="text-white " v-for="header in headerGroup.headers" :key="header.id">
                        <FlexRender v-if="!header.isPlaceholder" :render="header.column.columnDef.header"
                            :props="header.getContext()" />
                    </TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <template v-if="table.getRowModel().rows?.length">
                    <TableRow v-for="row in table.getRowModel().rows" :key="row.id"
                        :data-state="row.getIsSelected() ? 'selected' : undefined">
                        <TableCell v-for="cell in row.getVisibleCells()" :key="cell.id">
                            <FlexRender :render="cell.column.columnDef.cell" :props="cell.getContext()" />
                        </TableCell>
                    </TableRow>
                </template>
                <template v-else>
                    <TableRow>
                        <TableCell :colSpan="columns.length" class=" text-center h-16">
                            No results.
                        </TableCell>
                    </TableRow>
                </template>
            </TableBody>
        </Table></div>
        
       
    </div>

</template>