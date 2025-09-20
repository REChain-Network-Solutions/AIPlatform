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

namespace Tests\YooKassa\Request\PersonalData;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\PersonalData\PersonalDataType;
use YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest;
use YooKassa\Request\PersonalData\PersonalDataType\RecipientPersonalDataRequestFactory;

/**
 * RecipientPersonalDataRequestFactoryTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class RecipientPersonalDataRequestFactoryTest extends TestCase
{
    /**
     * @dataProvider validTypeDataProvider
     */
    public function testFactory(string $type): void
    {
        $instance = $this->getTestInstance();
        $paymentData = $instance->factory($type);
        self::assertNotNull($paymentData);
        self::assertInstanceOf(AbstractPersonalDataRequest::class, $paymentData);
        self::assertEquals($type, $paymentData->getType());
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
        $paymentData = $instance->factoryFromArray($options);
        self::assertNotNull($paymentData);
        self::assertInstanceOf(AbstractPersonalDataRequest::class, $paymentData);

        foreach ($options as $property => $value) {
            if ($paymentData->{$property} instanceof \DateTime) {
                self::assertEquals($paymentData->{$property}->format('Y-m-d'), $value);
            } else {
                self::assertEquals($paymentData->{$property}, $value);
            }
        }

        $type = $options['type'];
        unset($options['type']);
        $paymentData = $instance->factoryFromArray($options, $type);
        self::assertNotNull($paymentData);
        self::assertInstanceOf(AbstractPersonalDataRequest::class, $paymentData);

        self::assertEquals($type, $paymentData->getType());
        foreach ($options as $property => $value) {
            if ($paymentData->{$property} instanceof \DateTime) {
                self::assertEquals($paymentData->{$property}->format('Y-m-d'), $value);
            } else {
                self::assertEquals($paymentData->{$property}, $value);
            }
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

    /**
     * @return array
     */
    public static function validTypeDataProvider(): array
    {
        $result = [];
        foreach (PersonalDataType::getEnabledValues() as $value) {
            $result[] = [$value];
        }

        return $result;
    }

    /**
     * @return array
     * @throws \Exception
     */
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

    /**
     * @return \array[][]
     * @throws \Exception
     */
    public static function validArrayDataProvider(): array
    {
        $result = [
            [
                [
                    'type' => PersonalDataType::SBP_PAYOUT_RECIPIENT,
                    'first_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_FIRST_NAME, 'abcdefghijklmnopqrstuvwxyz -'),
                    'last_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, 'abcdefghijklmnopqrstuvwxyz -'),
                    'middle_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, 'abcdefghijklmnopqrstuvwxyz -'),
                ],
            ],
            [
                [
                    'type' => PersonalDataType::PAYOUT_STATEMENT_RECIPIENT,
                    'first_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_FIRST_NAME, 'abcdefghijklmnopqrstuvwxyz -'),
                    'last_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, 'abcdefghijklmnopqrstuvwxyz -'),
                    'middle_name' => Random::str(1, AbstractPersonalDataRequest::MAX_LENGTH_LAST_NAME, 'abcdefghijklmnopqrstuvwxyz -'),
                    'birthdate' => date('Y-m-d', Random::int(315619200, 946771200)),
                ],
            ],
        ];
        foreach (PersonalDataType::getEnabledValues() as $value) {
            $result[] = [['type' => $value]];
        }

        return $result;
    }

    /**
     * @return array
     */
    public static function invalidDataArrayDataProvider(): array
    {
        return [
            [[]],
            [['type' => 'test']],
        ];
    }

    /**
     * @return RecipientPersonalDataRequestFactory
     */
    protected function getTestInstance(): RecipientPersonalDataRequestFactory
    {
        return new RecipientPersonalDataRequestFactory();
    }
}
