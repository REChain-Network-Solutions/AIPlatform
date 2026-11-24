<?php

declare(strict_types=1);

namespace Database\Factories;

use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

class UserFactory extends Factory
{
    public function definition(): array
    {
        return [
            'name'           => fake()->name(),
            'surname'        => fake()->lastName(),
            'email'          => fake()->unique()->safeEmail(),
            'password'       => '$2y$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', // password,
            'entity_credits' => User::getFreshCredits(),
        ];
    }

    public function sharedCredit(float $balance = 1000.0): static
    {
        return $this->state(fn (array $attributes) => [
            'credit_system_type' => 'shared',
            'shared_credits'     => $balance,
        ]);
    }
}
