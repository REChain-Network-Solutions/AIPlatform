<?php

namespace Database\Factories;

use App\Extensions\PhoneCallAgent\System\Enums\TrainTypeEnum;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgent;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgentTrain;
use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

class ExtPhoneCallAgentTrainFactory extends Factory
{
    protected $model = ExtPhoneCallAgentTrain::class;

    public function definition(): array
    {
        return [
            'agent_id'   => ExtPhoneCallAgent::factory(),
            'user_id'    => User::factory(),
            'doc_id'     => null,
            'name'       => fake()->words(2, true),
            'type'       => TrainTypeEnum::text,
            'file'       => null,
            'url'        => null,
            'text'       => fake()->paragraph(),
            'trained_at' => null,
        ];
    }
}
