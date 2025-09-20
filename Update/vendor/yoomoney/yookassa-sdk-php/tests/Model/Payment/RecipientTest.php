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

namespace Tests\YooKassa\Model\Payment;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Recipient;

/**
 * RecipientTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class RecipientTest extends AbstractTestCase
{
    protected Recipient $object;

    /**
     * @return Recipient
     */
    protected function getTestInstance(): Recipient
    {
        return new Recipient();
    }

    /**
     * @return void
     */
    public function testRecipientClassExists(): void
    {
        $this->object = $this->getMockBuilder(Recipient::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Recipient::class));
        $this->assertInstanceOf(Recipient::class, $this->object);
    }

    /**
     * Test property "gateway_id"
     * @dataProvider validGatewayIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testGatewayId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getGatewayId());
        self::assertEmpty($instance->gateway_id);
        $instance->setGatewayId($value);
        self::assertEquals($value, is_array($value) ? $instance->getGatewayId()->toArray() : $instance->getGatewayId());
        self::assertEquals($value, is_array($value) ? $instance->gateway_id->toArray() : $instance->gateway_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getGatewayId());
            self::assertNotNull($instance->gateway_id);
        }
    }

    /**
     * Test invalid property "gateway_id"
     * @dataProvider invalidGatewayIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidGatewayId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setGatewayId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validGatewayIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_gateway_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidGatewayIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_gateway_id'));
    }
}
