<?php

namespace Database\Factories;

use App\Extensions\MarketingBot\System\Models\Whatsapp\Contact;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<Contact>
 */
class ContactFactory extends Factory
{
    /** @var class-string<Contact> */
    protected $model = Contact::class;

    /**
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        return [
            'user_id' => null,
            'name'    => $this->faker->word(),
            'status'  => true,
        ];
    }
}
