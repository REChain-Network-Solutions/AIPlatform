<?php

namespace Modules\WorkOrders\Database\Factories;

use Modules\WorkOrders\Entities\WorkOrder;
use Illuminate\Database\Eloquent\Factories\Factory;

class WorkOrderFactory extends Factory
{
    protected $model = WorkOrder::class;

    public function definition(): array
    {
        return [
            'client_id' => 1,
            'status' => $this->faker->randomElement(['open','in_progress','done']),
            'priority' => $this->faker->randomElement(['low','normal','high']),
            'scheduled_for' => now()->addDays(rand(0,10)),
            'due_by' => now()->addDays(rand(1,14)),
            'notes' => $this->faker->sentence(),
        ];
    }
}
