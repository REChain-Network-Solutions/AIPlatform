<?php

namespace Modules\PMCore\Database\Factories;

use Illuminate\Database\Eloquent\Factories\Factory;
use Modules\PMCore\app\Enums\ProjectPriority;
use Modules\PMCore\app\Enums\ProjectStatus;
use Modules\PMCore\app\Enums\ProjectType;
use Modules\PMCore\app\Models\Project;

class ProjectFactory extends Factory
{
    /**
     * The name of the factory's corresponding model.
     */
    protected $model = Project::class;

    /**
     * Define the model's default state.
     */
    public function definition(): array
    {
        $startDate = $this->faker->dateTimeBetween('-2 months', 'now');
        $endDate = $this->faker->dateTimeBetween($startDate, '+3 months');

        return [
            'name' => $this->faker->company().' '.$this->faker->randomElement(['Website', 'App', 'System', 'Platform']),
            'code' => strtoupper($this->faker->bothify('???-###')),
            'description' => $this->faker->paragraph(),
            'status' => $this->faker->randomElement(ProjectStatus::cases()),
            'type' => $this->faker->randomElement(ProjectType::cases()),
            'priority' => $this->faker->randomElement(ProjectPriority::cases()),
            'start_date' => $startDate,
            'end_date' => $endDate,
            'budget' => $this->faker->randomFloat(2, 10000, 100000),
            'hourly_rate' => $this->faker->randomFloat(2, 50, 200),
            'color_code' => $this->faker->hexColor(),
            'is_billable' => $this->faker->boolean(80),
            'project_manager_id' => null, // Set in seeder or test
        ];
    }

    /**
     * Indicate that the project is active.
     */
    public function active(): static
    {
        return $this->state(fn (array $attributes) => [
            'status' => ProjectStatus::IN_PROGRESS,
        ]);
    }

    /**
     * Indicate that the project is completed.
     */
    public function completed(): static
    {
        return $this->state(fn (array $attributes) => [
            'status' => ProjectStatus::COMPLETED,
        ]);
    }

    /**
     * Indicate that the project is billable.
     */
    public function billable(): static
    {
        return $this->state(fn (array $attributes) => [
            'is_billable' => true,
        ]);
    }

    /**
     * Indicate that the project is high priority.
     */
    public function highPriority(): static
    {
        return $this->state(fn (array $attributes) => [
            'priority' => ProjectPriority::HIGH,
        ]);
    }
}
