<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Model\Payment\Confirmation;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Payment\Confirmation\AbstractConfirmation;
use YooKassa\Model\Payment\Confirmation\ConfirmationFactory;
use YooKassa\Model\Payment\ConfirmationType;

/**
 * ConfirmationFactoryTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ConfirmationFactoryTest extends TestCase
{
    /**
     * @dataProvider validTypeDataProvider
     */
    public function testFactory(string $type): void
    {
        $instance = $this->getTestInstance();
        $confirmation = $instance->factory($type);
        self::assertNotNull($confirmation);
        self::assertInstanceOf(AbstractConfirmation::class, $confirmation);
        self::assertEquals($type, $confirmation->getType());
    }

    /**
     * @dataProvider invalidTypeDataProvider
     *
     * @param mixed $type
     */
    public function testInvalidFactory(mixed $type): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = $this->getTestInstance();
        $instance->factory($type);
    }

    /**
     * @dataProvider validArrayDataProvider
     */
    public function testFactoryFromArray(array $options): void
    {
        $instance = $this->getTestInstance();
        $confirmation = $instance->factoryFromArray($options);
        self::assertNotNull($confirmation);
        self::assertInstanceOf(AbstractConfirmation::class, $confirmation);

        foreach ($options as $property => $value) {
            self::assertEquals($confirmation->{$property}, $value);
        }

        $type = $options['type'];
        unset($options['type']);
        $confirmation = $instance->factoryFromArray($options, $type);
        self::assertNotNull($confirmation);
        self::assertInstanceOf(AbstractConfirmation::class, $confirmation);

        self::assertEquals($type, $confirmation->getType());
        foreach ($options as $property => $value) {
            self::assertEquals($confirmation->{$property}, $value);
        }
    }

    /**
     * @dataProvider invalidDataArrayDataProvider
     *
     * @param mixed $options
     */
    public function testInvalidFactoryFromArray(mixed $options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = $this->getTestInstance();
        $instance->factoryFromArray($options);
    }

    public static function validTypeDataProvider(): array
    {
        $result = [];
        foreach (ConfirmationType::getValidValues() as $value) {
            $result[] = [$value];
        }

        return $result;
    }

    public static function invalidTypeDataProvider(): array
    {
        return [
            [''],
            [0],
            [1],
            [-1],
            ['5'],
            [Random::str(10)],
        ];
    }

    public static function validArrayDataProvider(): array
    {
        $url = [
            'https://shop.store/testurl?pr=xXxXxX',
            'https://test.ru',
        ];
        $result = [
            [
                [
                    'type' => ConfirmationType::REDIRECT,
                    'enforce' => false,
                    'returnUrl' => Random::value($url),
                    'confirmationUrl' => Random::value($url),
                ],
            ],
            [
                [
                    'type' => ConfirmationType::REDIRECT,
                    'enforce' => true,
                    'returnUrl' => Random::value($url),
                ],
            ],
            [
                [
                    'type' => ConfirmationType::REDIRECT,
                    'returnUrl' => Random::value($url),
                    'confirmationUrl' => Random::value($url),
                ],
            ],
            [
                [
                    'type' => ConfirmationType::REDIRECT,
                    'confirmationUrl' => Random::value($url),
                ],
            ],
            [
                [
                    'type' => ConfirmationType::REDIRECT,
                    'returnUrl' => Random::value($url),
                ],
            ],
            [
                [
                    'type' => ConfirmationType::REDIRECT,
                    'enforce' => Random::bool(),
                ],
            ],
            [
                [
                    'type' => ConfirmationType::REDIRECT,
                ],
            ],
            [
                [
                    'type' => ConfirmationType::QR,
                    'confirmation_data' => Random::str(30),
                ],
            ],
        ];
        foreach (ConfirmationType::getValidValues() as $value) {
            $result[] = [['type' => $value]];
        }

        return $result;
    }

    public static function invalidDataArrayDataProvider(): array
    {
        return [
            [[]],
            [['type' => 'test']],
        ];
    }

    protected function getTestInstance(): ConfirmationFactory
    {
        return new ConfirmationFactory();
    }
}
