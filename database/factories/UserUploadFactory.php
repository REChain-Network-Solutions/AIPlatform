<?php

namespace Database\Factories;

use App\Models\User;
use App\Models\UserUpload;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<UserUpload>
 */
class UserUploadFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        return [
            'user_id'       => User::factory(),
            'disk'          => 'local',
            'path'          => 'user-uploads/' . fake()->uuid() . '.pdf',
            'original_name' => 'document.pdf',
            'mime_type'     => 'application/pdf',
            'extension'     => 'pdf',
            'size'          => 100,
            'sha256'        => hash('sha256', fake()->uuid()),
        ];
    }
}
