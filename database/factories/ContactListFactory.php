<?php

namespace Database\Factories;

use App\Extensions\MarketingBot\System\Models\Whatsapp\ContactList;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<ContactList>
 */
class ContactListFactory extends Factory
{
    /** @var class-string<ContactList> */
    protected $model = ContactList::class;

    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        return [
            'user_id'      => null,
            'name'         => $this->faker->name(),
            'phone'        => '+1' . $this->faker->numerify('##########'),
            'country_code' => 1,
            'avatar'       => null,
        ];
    }
}
