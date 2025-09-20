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

namespace Tests\YooKassa\Model\Notification;

use Exception;
use InvalidArgumentException;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Notification\NotificationSucceeded;
use YooKassa\Model\Receipt\SettlementType;

/**
 * NotificationSucceededTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class NotificationSucceededTest extends AbstractTestCase
{
    protected NotificationSucceeded $object;

    /**
     * @param mixed|null $value
     * @return NotificationSucceeded
     */
    protected function getTestInstance(mixed $value = null): NotificationSucceeded
    {
        return new NotificationSucceeded($value);
    }

    /**
     * @return void
     */
    public function testNotificationCanceledClassExists(): void
    {
        $this->object = $this->getMockBuilder(NotificationSucceeded::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(NotificationSucceeded::class));
        $this->assertInstanceOf(NotificationSucceeded::class, $this->object);
    }

    /**
     * Test property "object"
     * @dataProvider validObjectDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setObject($value);
        self::assertNotNull($instance->getObject());
        self::assertNotNull($instance->object);
        self::assertEquals($value, is_array($value) ? $instance->getObject()->toArray() : $instance->getObject());
        self::assertEquals($value, is_array($value) ? $instance->object->toArray() : $instance->object);
    }

    /**
     * Test invalid property "object"
     * @dataProvider invalidObjectDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidObject(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setObject($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validObjectDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_object'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidObjectDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_object'));
    }

    /**
     * Test valid method "fromArray"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testFromArray(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->fromArray($value->toArray());
        self::assertEquals($value['object'], $instance->getObject());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        $objects = $this->validObjectDataProvider();
        $instance->setObject(array_shift($objects[0]));
        return [[$instance]];
    }

    /**
     * @dataProvider invalidDataProvider
     */
    public function testInvalidFromArray(array $options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->getTestInstance($options);
    }

    /**
     * @return \array[][]
     */
    public static function invalidDataProvider(): array
    {
        return [
            [
                [
                    'type' => SettlementType::PREPAYMENT,
                ],
            ],
            [
                [
                    'event' => SettlementType::PREPAYMENT,
                ],
            ],
            [
                [
                    'object' => [],
                ],
            ],
        ];
    }
}
