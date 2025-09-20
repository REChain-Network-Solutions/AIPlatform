<?php

namespace Tests\YooKassa\Validator\Constraints;

use PHPUnit\Framework\TestCase;
use YooKassa\Validator\Constraints\NotNull;

class NotNullTest extends TestCase
{
    private function getInstance(mixed $message): NotNull
    {
        return new NotNull($message);
    }

    /**
     * @dataProvider validDataProvider
     * @param array $options
     * @return void
     */
    public function testGetMessage(array $options): void
    {
        $message = $options['message'];

        $instance = $this->getInstance($message);

        if ($message === null) {
            $this->assertNotEmpty($instance->getMessage());
        } else {
            $this->assertSame($message, $instance->getMessage());
        }

    }

    public function validDataProvider(): array
    {
        return [
            [['message' => 'Some error message']],
            [['message' => null]],
        ];
    }
}
