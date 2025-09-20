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

namespace Tests\YooKassa\Model\Payment\PaymentMethod;

use Exception;
use InvalidArgumentException;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod;

/**
 * AbstractTestPaymentMethod
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
abstract class AbstractTestPaymentMethod extends AbstractTestCase
{
    public function testGetType(): void
    {
        $instance = $this->getTestInstance();
        self::assertEquals($this->getExpectedType(), $instance->getType());
    }

    /**
     * @dataProvider invalidTypeDataProvider
     */
    public function testInvalidType(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        new TestPaymentData($value);
    }

    /**
     * @dataProvider validSavedDataProvider
     */
    public function testGetSetSaved(mixed $value): void
    {
        $instance = $this->getTestInstance();

        self::assertFalse($instance->getSaved());
        self::assertFalse($instance->saved);

        $instance->setSaved($value);
        if ($value) {
            self::assertTrue($instance->getSaved());
            self::assertTrue($instance->saved);
        } else {
            self::assertFalse($instance->getSaved());
            self::assertFalse($instance->saved);
        }

        $instance = $this->getTestInstance();
        $instance->saved = $value;
        if ($value) {
            self::assertTrue($instance->getSaved());
            self::assertTrue($instance->saved);
        } else {
            self::assertFalse($instance->getSaved());
            self::assertFalse($instance->saved);
        }
    }

    /**
     * Test property "status"
     * @dataProvider validStatusDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testStatus(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setStatus($value);
        self::assertNotNull($instance->getStatus());
        self::assertNotNull($instance->status);
        self::assertEquals($value, $instance->getStatus());
        self::assertEquals($value, $instance->status);
    }

    /**
     * Test invalid property "status"
     * @dataProvider invalidStatusDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidStatus(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setStatus($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
     * @dataProvider validIdDataProvider
     */
    public function testGetSetId(mixed $value): void
    {
        $instance = $this->getTestInstance();

        $instance->setId($value);
        if (empty($value)) {
            self::assertNull($instance->getId());
            self::assertNull($instance->id);
        } else {
            self::assertEquals($value, $instance->getId());
            self::assertEquals($value, $instance->id);
        }

        $instance = $this->getTestInstance();
        $instance->id = $value;
        if (empty($value)) {
            self::assertNull($instance->getId());
            self::assertNull($instance->id);
        } else {
            self::assertEquals($value, $instance->getId());
            self::assertEquals($value, $instance->id);
        }
    }

    /**
     * @dataProvider validTitleDataProvider
     */
    public function testGetSetTitle(mixed $value): void
    {
        $instance = $this->getTestInstance();

        $instance->setTitle($value);
        if (empty($value)) {
            self::assertNull($instance->getTitle());
            self::assertNull($instance->title);
        } else {
            self::assertEquals($value, $instance->getTitle());
            self::assertEquals($value, $instance->title);
        }

        $instance = $this->getTestInstance();
        $instance->title = $value;
        if (empty($value)) {
            self::assertNull($instance->getTitle());
            self::assertNull($instance->title);
        } else {
            self::assertEquals($value, $instance->getTitle());
            self::assertEquals($value, $instance->title);
        }
    }

    public static function invalidTypeDataProvider(): array
    {
        return [
            [''],
            [null],
            [Random::str(40)],
            [0],
        ];
    }

    public static function validSavedDataProvider(): array
    {
        return [
            [true],
            [false],
        ];
    }

    public static function validIdDataProvider(): array
    {
        return [
            [null],
            [Random::str(2)],
            [Random::str(10)],
            [Random::str(100)],
        ];
    }

    public static function validTitleDataProvider(): array
    {
        return [
            [null],
            [Random::str(2, 2, '123456789ABCDEF')],
            [Random::str(2)],
            [Random::str(10)],
            [Random::str(100)],
        ];
    }

    abstract protected function getTestInstance(): AbstractPaymentMethod;

    abstract protected function getExpectedType(): string;
}

class TestPaymentData extends AbstractPaymentMethod
{
    public function __construct($type)
    {
        parent::__construct([]);
        $this->setType($type);
    }
}
