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

namespace Tests\YooKassa\Request\Payments\ConfirmationAttributes;

use Exception;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes;

/**
 * AbstractTestConfirmationAttributes
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
abstract class AbstractTestConfirmationAttributes extends TestCase
{
    public function testGetType(): void
    {
        $instance = $this->getTestInstance();
        self::assertEquals($this->getExpectedType(), $instance->getType());
    }

    /**
     * @dataProvider invalidTypeDataProvider
     *
     * @param mixed $value
     */
    public function testInvalidType(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        new TestConfirmation($value);
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

    /**
     * @dataProvider validLocaleDataProvider
     *
     * @param mixed $value
     */
    public function testSetterLocale(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setLocale($value);
        self::assertEquals((string) $value, $instance->getLocale());
    }

    /**
     * @throws Exception
     */
    public static function validLocaleDataProvider(): array
    {
        return [
            [''],
            [null],
            ['ru_RU'],
            ['en_US'],
        ];
    }

    /**
     * @dataProvider invalidLocaleDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidLocale(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->getTestInstance()->setLocale($value);
    }

    /**
     * @throws Exception
     */
    public static function invalidLocaleDataProvider(): array
    {
        return [
            [Random::str(4)],
            [Random::str(6)],
            [0],
        ];
    }

    abstract protected function getTestInstance(): AbstractConfirmationAttributes;

    abstract protected function getExpectedType(): string;
}

class TestConfirmation extends AbstractConfirmationAttributes
{
    public function __construct($type)
    {
        parent::__construct();
        $this->setType($type);
    }
}
