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

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Deal\DealInterface;
use YooKassa\Model\Notification\AbstractNotification;
use YooKassa\Model\Payment\PaymentInterface;
use YooKassa\Model\Payout\PayoutInterface;
use YooKassa\Model\Refund\RefundInterface;

/**
 * AbstractTestNotification
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
abstract class AbstractTestNotification extends TestCase
{
    abstract public function validDataProvider(): array;

    /**
     * @dataProvider validDataProvider
     */
    public function testGetType(array $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertEquals($this->getExpectedType(), $instance->getType());
    }

    /**
     * @dataProvider invalidConstructorTypeDataProvider
     */
    public function testInvalidTypeInConstructor(array $source): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->getTestInstance($source);
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetEvent(array $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertEquals($this->getExpectedEvent(), $instance->getEvent());
    }

    /**
     * @dataProvider invalidConstructorEventDataProvider
     */
    public function testInvalidEventInConstructor(array $source): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->getTestInstance($source);
    }

    /**
     * @dataProvider invalidTypeDataProvider
     *
     * @param mixed $value
     */
    public function testInvalidType($value): void
    {
        $this->expectException(InvalidArgumentException::class);
        new TestNotification($value, $this->getExpectedEvent());
    }

    /**
     * @dataProvider invalidEventDataProvider
     *
     * @param mixed $value
     */
    public function testInvalidEvent($value): void
    {
        $this->expectException(InvalidArgumentException::class);
        new TestNotification($this->getExpectedType(), $value);
    }

    public function invalidConstructorTypeDataProvider()
    {
        return [
            [['event' => $this->getExpectedEvent(), 'type' => 'test']],
            [['event' => $this->getExpectedEvent(), 'type' => null]],
            [['event' => $this->getExpectedEvent(), 'type' => '']],
            [['event' => $this->getExpectedEvent(), 'type' => 1]],
            [['event' => $this->getExpectedEvent(), 'type' => []]],
        ];
    }

    public function invalidConstructorEventDataProvider()
    {
        return [
            [['type' => $this->getExpectedType(), 'event' => 'test']],
            [['type' => $this->getExpectedType(), 'event' => null]],
            [['type' => $this->getExpectedType(), 'event' => '']],
            [['type' => $this->getExpectedType(), 'event' => 1]],
            [['type' => $this->getExpectedType(), 'event' => []]],
        ];
    }

    public static function invalidTypeDataProvider()
    {
        return [
            [''],
            [null],
            [Random::str(40)],
            [0],
        ];
    }

    public static function invalidEventDataProvider()
    {
        return [
            [''],
            [null],
            [Random::str(40)],
            [0],
        ];
    }

    abstract protected function getTestInstance(array $source): AbstractNotification;

    abstract protected function getExpectedType(): string;

    abstract protected function getExpectedEvent(): string;
}

class TestNotification extends AbstractNotification
{
    public function __construct($type, $event)
    {
        parent::__construct();
        $this->setType($type);
        $this->setEvent($event);
    }

    public function getObject(): PaymentInterface|RefundInterface|PayoutInterface|DealInterface|null
    {
        return null;
    }
}
