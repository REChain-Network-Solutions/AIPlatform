<?php

namespace Modules\PMCore\database\factories;

use App\Models\User;
use Carbon\Carbon;
use Illuminate\Database\Eloquent\Factories\Factory;
use Modules\PMCore\app\Enums\TimesheetStatus;
use Modules\PMCore\app\Models\Project;
use Modules\PMCore\app\Models\Timesheet;

class TimesheetFactory extends Factory
{
    /**
     * The name of the factory's corresponding model.
     *
     * @var string
     */
    protected $model = Timesheet::class;

    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        $date = $this->faker->dateTimeBetween('-30 days', 'today');
        $hours = $this->faker->randomFloat(2, 0.5, 8);
        $isBillable = $this->faker->boolean(80); // 80% chance of being billable
        $billingRate = $isBillable ? $this->faker->numberBetween(50, 150) : null;
        $costRate = $billingRate ? round($billingRate * ($this->faker->numberBetween(40, 60) / 100), 2) : null;

        $descriptions = [
            'Worked on implementing new features',
            'Fixed bugs and improved performance',
            'Attended meetings and updated documentation',
            'Code review and refactoring',
            'Implemented API endpoints',
            'Database optimization',
            'Created unit tests',
            'Updated user interface',
            'Deployment and configuration',
            'Research and development',
        ];

        return [
            'user_id' => User::inRandomOrder()->first()?->id ?? User::factory(),
            'project_id' => Project::inRandomOrder()->first()?->id ?? Project::factory(),
            'task_id' => null, // Can be set explicitly when needed
            'date' => $date,
            'hours' => $hours,
            'description' => $this->faker->randomElement($descriptions).' - '.$this->faker->sentence(),
            'is_billable' => $isBillable,
            'billing_rate' => $billingRate,
            'cost_rate' => $costRate,
            'status' => $this->faker->randomElement(TimesheetStatus::cases())->value,
            'approved_by_id' => null,
            'approved_at' => null,
            'created_by_id' => function (array $attributes) {
                return $attributes['user_id'];
            },
            'updated_by_id' => function (array $attributes) {
                return $attributes['user_id'];
            },
        ];
    }

    /**
     * Indicate that the timesheet is in draft status.
     */
    public function draft(): static
    {
        return $this->state(fn (array $attributes) => [
            'status' => TimesheetStatus::DRAFT->value,
            'approved_by_id' => null,
            'approved_at' => null,
        ]);
    }

    /**
     * Indicate that the timesheet is submitted.
     */
    public function submitted(): static
    {
        return $this->state(fn (array $attributes) => [
            'status' => TimesheetStatus::SUBMITTED->value,
            'approved_by_id' => null,
            'approved_at' => null,
        ]);
    }

    /**
     * Indicate that the timesheet is approved.
     */
    public function approved(): static
    {
        return $this->state(fn (array $attributes) => [
            'status' => TimesheetStatus::APPROVED->value,
            'approved_by_id' => User::where('id', '!=', $attributes['user_id'])->inRandomOrder()->first()?->id,
            'approved_at' => Carbon::parse($attributes['date'])->addDay()->setTime(rand(9, 18), rand(0, 59)),
        ]);
    }

    /**
     * Indicate that the timesheet is rejected.
     */
    public function rejected(): static
    {
        return $this->state(fn (array $attributes) => [
            'status' => TimesheetStatus::REJECTED->value,
            'approved_by_id' => User::where('id', '!=', $attributes['user_id'])->inRandomOrder()->first()?->id,
            'approved_at' => Carbon::parse($attributes['date'])->addDay()->setTime(rand(9, 18), rand(0, 59)),
        ]);
    }

    /**
     * Indicate that the timesheet is invoiced.
     */
    public function invoiced(): static
    {
        return $this->state(fn (array $attributes) => [
            'status' => TimesheetStatus::INVOICED->value,
            'approved_by_id' => User::where('id', '!=', $attributes['user_id'])->inRandomOrder()->first()?->id,
            'approved_at' => Carbon::parse($attributes['date'])->addDay()->setTime(rand(9, 18), rand(0, 59)),
        ]);
    }

    /**
     * Indicate that the timesheet is billable.
     */
    public function billable(): static
    {
        return $this->state(fn (array $attributes) => [
            'is_billable' => true,
            'billing_rate' => $this->faker->numberBetween(50, 150),
            'cost_rate' => function (array $attributes) {
                return round($attributes['billing_rate'] * ($this->faker->numberBetween(40, 60) / 100), 2);
            },
        ]);
    }

    /**
     * Indicate that the timesheet is non-billable.
     */
    public function nonBillable(): static
    {
        return $this->state(fn (array $attributes) => [
            'is_billable' => false,
            'billing_rate' => null,
            'cost_rate' => null,
        ]);
    }

    /**
     * Configure the model factory.
     */
    public function configure(): static
    {
        return $this->afterMaking(function (Timesheet $timesheet) {
            // Additional logic after making
        })->afterCreating(function (Timesheet $timesheet) {
            // Additional logic after creating
        });
    }
}
