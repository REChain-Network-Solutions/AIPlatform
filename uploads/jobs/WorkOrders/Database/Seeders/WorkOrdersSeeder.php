<?php

namespace Modules\WorkOrders\Database\Seeders;

use Illuminate\Database\Seeder;
use Modules\WorkOrders\Entities\{WorkOrder, WOType, WOServiceAppointment, WOServiceTask, WOServicePart, ServiceTask, ServicePart};

class WorkOrdersSeeder extends Seeder
{
    public function run(): void
    {
        // Catalog
        $st = ServiceTask::firstOrCreate(['sku' => 'LAB-001'], ['name' => 'Labor Hour', 'default_rate' => 120]);
        $sp = ServicePart::firstOrCreate(['sku' => 'PART-001'], ['name' => 'Generic Part', 'sale_price' => 45]);

        // Work Order
        $wo = WorkOrder::create([
            'client_id' => 1,
            'status' => 'open',
            'priority' => 'normal',
            'scheduled_for' => now()->addDay(),
            'notes' => 'Demo work order',
        ]);

        // Lines
        WOServiceTask::create([ 'work_order_id' => $wo->id, 'service_task_id' => $st->id, 'qty' => 2, 'rate' => 120, 'total' => 240 ]);
        WOServicePart::create([ 'work_order_id' => $wo->id, 'service_part_id' => $sp->id, 'qty' => 1, 'price' => 45, 'total' => 45 ]);

        // Appointment
        WOServiceAppointment::create([
            'work_order_id' => $wo->id,
            'technician_id' => 1,
            'starts_at' => now()->addDays(2)->setTime(9,0),
            'ends_at' => now()->addDays(2)->setTime(11,0),
            'status' => 'scheduled',
        ]);
    }
}
