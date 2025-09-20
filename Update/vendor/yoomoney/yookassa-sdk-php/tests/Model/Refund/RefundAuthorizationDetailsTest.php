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

namespace Tests\YooKassa\Model\Refund;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Refund\RefundAuthorizationDetails;

/**
 * RefundAuthorizationDetailsTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class RefundAuthorizationDetailsTest extends AbstractTestCase
{
    protected RefundAuthorizationDetails $object;

    /**
    * @return RefundAuthorizationDetails
    */
    protected function getTestInstance(): RefundAuthorizationDetails
    {
        return new RefundAuthorizationDetails();
    }

    /**
    * @return void
    */
    public function testRefundAuthorizationDetailsClassExists(): void
    {
        $this->object = $this->getMockBuilder(RefundAuthorizationDetails::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(RefundAuthorizationDetails::class));
        $this->assertInstanceOf(RefundAuthorizationDetails::class, $this->object);
    }

    /**
    * Test property "rrn"
    * @dataProvider validRrnDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testRrn(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getRrn());
        self::assertEmpty($instance->rrn);
        $instance->setRrn($value);
        self::assertEquals($value, is_array($value) ? $instance->getRrn()->toArray() : $instance->getRrn());
        self::assertEquals($value, is_array($value) ? $instance->rrn->toArray() : $instance->rrn);
        if (!empty($value)) {
            self::assertNotNull($instance->getRrn());
            self::assertNotNull($instance->rrn);
        }
    }

    /**
    * Test invalid property "rrn"
    * @dataProvider invalidRrnDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidRrn(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRrn($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validRrnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_rrn'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidRrnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_rrn'));
    }
}
