<?php

namespace Modules\WorkOrders\Database\Factories;

use Modules\WorkOrders\Entities\ServiceTask;
use Illuminate\Database\Eloquent\Factories\Factory;

class ServiceTaskFactory extends Factory
{
    protected $model = ServiceTask::class;

    public function definition(): array
    {
        return [
            'sku' => 'LAB-'.strtoupper($this->faker->bothify('???###')),
            'name' => $this->faker->words(3, true),
            'default_rate' => $this->faker->randomFloat(2, 80, 180),
        ];
    }
}
