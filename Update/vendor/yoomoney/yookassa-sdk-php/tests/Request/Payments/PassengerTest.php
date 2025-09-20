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

namespace Tests\YooKassa\Request\Payments;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\Passenger;

/**
 * AirlinePassengerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PassengerTest extends AbstractTestCase
{
    protected Passenger $object;

    /**
     * @return Passenger
     */
    protected function getTestInstance(): Passenger
    {
        return new Passenger();
    }

    /**
     * @return void
     */
    public function testPassengerClassExists(): void
    {
        $this->object = $this->getMockBuilder(Passenger::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Passenger::class));
        $this->assertInstanceOf(Passenger::class, $this->object);
    }

    /**
     * Test property "first_name"
     * @dataProvider validFirstNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFirstName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setFirstName($value);
        self::assertNotNull($instance->getFirstName());
        self::assertNotNull($instance->first_name);
        self::assertEquals($value, is_array($value) ? $instance->getFirstName()->toArray() : $instance->getFirstName());
        self::assertEquals($value, is_array($value) ? $instance->first_name->toArray() : $instance->first_name);
        self::assertLessThanOrEqual(64, is_string($instance->getFirstName()) ? mb_strlen($instance->getFirstName()) : $instance->getFirstName());
        self::assertLessThanOrEqual(64, is_string($instance->first_name) ? mb_strlen($instance->first_name) : $instance->first_name);
        self::assertGreaterThanOrEqual(1, is_string($instance->getFirstName()) ? mb_strlen($instance->getFirstName()) : $instance->getFirstName());
        self::assertGreaterThanOrEqual(1, is_string($instance->first_name) ? mb_strlen($instance->first_name) : $instance->first_name);
    }

    /**
     * Test invalid property "first_name"
     * @dataProvider invalidFirstNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFirstName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFirstName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFirstNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_first_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFirstNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_first_name'));
    }

    /**
     * Test property "last_name"
     * @dataProvider validLastNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testLastName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setLastName($value);
        self::assertNotNull($instance->getLastName());
        self::assertNotNull($instance->last_name);
        self::assertEquals($value, is_array($value) ? $instance->getLastName()->toArray() : $instance->getLastName());
        self::assertEquals($value, is_array($value) ? $instance->last_name->toArray() : $instance->last_name);
        self::assertLessThanOrEqual(64, is_string($instance->getLastName()) ? mb_strlen($instance->getLastName()) : $instance->getLastName());
        self::assertLessThanOrEqual(64, is_string($instance->last_name) ? mb_strlen($instance->last_name) : $instance->last_name);
        self::assertGreaterThanOrEqual(1, is_string($instance->getLastName()) ? mb_strlen($instance->getLastName()) : $instance->getLastName());
        self::assertGreaterThanOrEqual(1, is_string($instance->last_name) ? mb_strlen($instance->last_name) : $instance->last_name);
    }

    /**
     * Test invalid property "last_name"
     * @dataProvider invalidLastNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidLastName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setLastName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validLastNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_last_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidLastNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_last_name'));
    }
}
