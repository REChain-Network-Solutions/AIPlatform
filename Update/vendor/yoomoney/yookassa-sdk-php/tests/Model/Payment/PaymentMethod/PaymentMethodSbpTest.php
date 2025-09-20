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
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\PaymentMethodSbp;

/**
 * PaymentMethodSbpTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentMethodSbpTest extends AbstractTestCase
{
    protected PaymentMethodSbp $object;

    /**
     * @return PaymentMethodSbp
     */
    protected function getTestInstance(): PaymentMethodSbp
    {
        return new PaymentMethodSbp();
    }

    /**
     * @return void
     */
    public function testPaymentMethodSbpClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentMethodSbp::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentMethodSbp::class));
        $this->assertInstanceOf(PaymentMethodSbp::class, $this->object);
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
        self::assertContains($instance->getType(), ['sbp']);
        self::assertContains($instance->type, ['sbp']);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
    }

    /**
     * Test invalid property "type"
     * @dataProvider invalidTypeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setType($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
     * Test property "sbp_operation_id"
     * @dataProvider validSbpOperationIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSbpOperationId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSbpOperationId());
        self::assertEmpty($instance->sbp_operation_id);
        $instance->setSbpOperationId($value);
        self::assertEquals($value, $instance->getSbpOperationId());
        self::assertEquals($value, $instance->sbp_operation_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getSbpOperationId());
            self::assertNotNull($instance->sbp_operation_id);
        }
    }

    /**
     * Test invalid property "sbp_operation_id"
     * @dataProvider invalidSbpOperationIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSbpOperationId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSbpOperationId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSbpOperationIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_sbp_operation_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSbpOperationIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_sbp_operation_id'));
    }

    /**
     * Test property "payer_bank_details"
     * @dataProvider validPayerBankDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPayerBankDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPayerBankDetails());
        self::assertEmpty($instance->payer_bank_details);
        $instance->setPayerBankDetails($value);
        self::assertEquals($value, is_array($value) ? $instance->getPayerBankDetails()->toArray() : $instance->getPayerBankDetails());
        self::assertEquals($value, is_array($value) ? $instance->payer_bank_details->toArray() : $instance->payer_bank_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getPayerBankDetails());
            self::assertNotNull($instance->payer_bank_details);
        }
    }

    /**
     * Test invalid property "payer_bank_details"
     * @dataProvider invalidPayerBankDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPayerBankDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPayerBankDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPayerBankDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payer_bank_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPayerBankDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payer_bank_details'));
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
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_params_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidQrcIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_params_id'));
    }

    /**
     * Test property "params_id"
     * @dataProvider validQrcIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testParamsId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getParamsId());
        self::assertEmpty($instance->paramsId);
        self::assertEmpty($instance->params_id);
        $instance->setParamsId($value);
        self::assertEquals($value, $instance->getParamsId());
        self::assertEquals($value, $instance->params_id);
        self::assertEquals($value, $instance->paramsId);
        if (!empty($value)) {
            self::assertNotNull($instance->getParamsId());
            self::assertNotNull($instance->paramsId);
            self::assertNotNull($instance->params_id);
            self::assertMatchesRegularExpression("/[A-Za-z0-9]{32}/", $instance->getParamsId());
            self::assertMatchesRegularExpression("/[A-Za-z0-9]{32}/", $instance->paramsId);
            self::assertMatchesRegularExpression("/[A-Za-z0-9]{32}/", $instance->params_id);
        }
    }

    /**
     * Test invalid property "params_id"
     * @dataProvider invalidParamsIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidParamsId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setParamsId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validParamsIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_params_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidParamsIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_params_id'));
    }

}
