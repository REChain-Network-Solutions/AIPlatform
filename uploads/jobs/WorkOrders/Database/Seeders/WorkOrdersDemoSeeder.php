<?php

namespace Modules\WorkOrders\Database\Seeders;

use Illuminate\Database\Seeder;
use Modules\WorkOrders\Entities\{WorkOrder, WOServiceTask, WOServicePart, ServiceTask, ServicePart};

class WorkOrdersDemoSeeder extends Seeder
{
    public function run(): void
    {
        // Catalog
        $tasks = ServiceTask::factory()->count(5)->create();
        $parts = ServicePart::factory()->count(5)->create();

        // Work Orders
        $orders = WorkOrder::factory()->count(10)->create();

        foreach ($orders as $wo) {
            // Attach 1-3 tasks
            foreach ($tasks->random(rand(1,3)) as $t) {
                $qty = rand(1,3);
                $rate = $t->default_rate ?? 100;
                WOServiceTask::create([
                    'work_order_id' => $wo->id,
                    'service_task_id' => $t->id,
                    'qty' => $qty,
                    'rate' => $rate,
                    'total' => $qty * $rate,
                ]);
            }
            // Attach 1-2 parts
            foreach ($parts->random(rand(1,2)) as $p) {
                $qty = rand(1,2);
                $price = $p->sale_price ?? 50;
                WOServicePart::create([
                    'work_order_id' => $wo->id,
                    'service_part_id' => $p->id,
                    'qty' => $qty,
                    'price' => $price,
                    'total' => $qty * $price,
                ]);
            }
        }
    }
}
