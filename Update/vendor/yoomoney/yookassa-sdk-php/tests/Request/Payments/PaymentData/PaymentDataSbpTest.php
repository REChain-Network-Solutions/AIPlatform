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

namespace Tests\YooKassa\Request\Payments\PaymentData;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentData\PaymentDataSbp;

/**
 * PaymentDataSbpTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentDataSbpTest extends AbstractTestCase
{
    protected PaymentDataSbp $object;

    /**
     * @return PaymentDataSbp
     */
    protected function getTestInstance(): PaymentDataSbp
    {
        return new PaymentDataSbp();
    }

    /**
     * @return void
     */
    public function testPaymentDataSbpClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentDataSbp::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentDataSbp::class));
        $this->assertInstanceOf(PaymentDataSbp::class, $this->object);
    }

    /**
     * Test property "type"
     *
     * @return void
     * @throws Exception
     */
    public function testType(): void
    {
        $instance = $this->getTestInstance();
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertContains($instance->getType(), ['sbp']);
        self::assertContains($instance->type, ['sbp']);
    }

    /**
     * Test property "qrc_id"
     * @dataProvider validQrcIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testQrcId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getQrcId());
        self::assertEmpty($instance->qrcId);
        self::assertEmpty($instance->qrc_id);
        $instance->setQrcId($value);
        self::assertEquals($value, $instance->getQrcId());
        self::assertEquals($value, $instance->qrc_id);
        self::assertEquals($value, $instance->qrcId);
        if (!empty($value)) {
            self::assertNotNull($instance->getQrcId());
            self::assertNotNull($instance->qrcId);
            self::assertNotNull($instance->qrc_id);
            self::assertMatchesRegularExpression("/[A-Za-z0-9]{32}/", $instance->getQrcId());
            self::assertMatchesRegularExpression("/[A-Za-z0-9]{32}/", $instance->qrcId);
            self::assertMatchesRegularExpression("/[A-Za-z0-9]{32}/", $instance->qrc_id);
        }
    }

    /**
     * Test invalid property "qrc_id"
     * @dataProvider invalidQrcIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidQrcId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setQrcId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validQrcIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_qrc_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidQrcIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_qrc_id'));
    }

    /**
     * Test property "expires_at"
     * @dataProvider validExpiresAtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiresAt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getQrcId());
        self::assertEmpty($instance->expiresAt);
        self::assertEmpty($instance->expires_at);
        $instance->setExpiresAt($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getExpiresAt());
            self::assertNotNull($instance->expiresAt);
            self::assertNotNull($instance->expires_at);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getExpiresAt());
                self::assertEquals($value, $instance->expiresAt);
                self::assertEquals($value, $instance->expires_at);
            } else {
                self::assertEquals(new Datetime($value), $instance->getExpiresAt());
                self::assertEquals(new Datetime($value), $instance->expiresAt);
                self::assertEquals(new Datetime($value), $instance->expires_at);
            }
        }
    }

    /**
     * Test invalid property "expires_at"
     * @dataProvider invalidExpiresAtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiresAt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiresAt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiresAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiresAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at'));
    }

}
