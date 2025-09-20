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

namespace Tests\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate;

/**
 * ElectronicCertificateTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class ElectronicCertificateTest extends AbstractTestCase
{
    protected ElectronicCertificate $object;

    /**
    * @return ElectronicCertificate
    */
    protected function getTestInstance(): ElectronicCertificate
    {
        return new ElectronicCertificate();
    }

    /**
    * @return void
    */
    public function testElectronicCertificateClassExists(): void
    {
        $this->object = $this->getMockBuilder(ElectronicCertificate::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ElectronicCertificate::class));
        $this->assertInstanceOf(ElectronicCertificate::class, $this->object);
    }

    /**
    * Test property "certificate_id"
    * @dataProvider validCertificateIdDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testCertificateId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setCertificateId($value);
        self::assertNotNull($instance->getCertificateId());
        self::assertNotNull($instance->certificate_id);
        self::assertEquals($value, $instance->getCertificateId());
        self::assertEquals($value, $instance->certificate_id);
        self::assertLessThanOrEqual(30, is_string($instance->getCertificateId()) ? mb_strlen($instance->getCertificateId()) : $instance->getCertificateId());
        self::assertLessThanOrEqual(30, is_string($instance->certificate_id) ? mb_strlen($instance->certificate_id) : $instance->certificate_id);
        self::assertGreaterThanOrEqual(20, is_string($instance->getCertificateId()) ? mb_strlen($instance->getCertificateId()) : $instance->getCertificateId());
        self::assertGreaterThanOrEqual(20, is_string($instance->certificate_id) ? mb_strlen($instance->certificate_id) : $instance->certificate_id);
    }

    /**
    * Test invalid property "certificate_id"
    * @dataProvider invalidCertificateIdDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidCertificateId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCertificateId($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validCertificateIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_certificate_id'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidCertificateIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_certificate_id'));
    }

    /**
    * Test property "tru_quantity"
    * @dataProvider validTruQuantityDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testTruQuantity(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setTruQuantity($value);
        self::assertNotNull($instance->getTruQuantity());
        self::assertNotNull($instance->tru_quantity);
        self::assertEquals($value, $instance->getTruQuantity());
        self::assertEquals($value, $instance->tru_quantity);
        self::assertIsNumeric($instance->getTruQuantity());
        self::assertIsNumeric($instance->tru_quantity);
    }

    /**
    * Test invalid property "tru_quantity"
    * @dataProvider invalidTruQuantityDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidTruQuantity(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTruQuantity($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validTruQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_tru_quantity'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidTruQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_tru_quantity'));
    }

    /**
    * Test property "available_compensation"
    * @dataProvider validAvailableCompensationDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testAvailableCompensation(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAvailableCompensation($value);
        self::assertNotNull($instance->getAvailableCompensation());
        self::assertNotNull($instance->available_compensation);
        self::assertEquals($value, is_array($value) ? $instance->getAvailableCompensation()->toArray() : $instance->getAvailableCompensation());
        self::assertEquals($value, is_array($value) ? $instance->available_compensation->toArray() : $instance->available_compensation);
    }

    /**
    * Test invalid property "available_compensation"
    * @dataProvider invalidAvailableCompensationDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidAvailableCompensation(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAvailableCompensation($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validAvailableCompensationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_available_compensation'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidAvailableCompensationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_available_compensation'));
    }

    /**
    * Test property "applied_compensation"
    * @dataProvider validAppliedCompensationDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testAppliedCompensation(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAppliedCompensation($value);
        self::assertNotNull($instance->getAppliedCompensation());
        self::assertNotNull($instance->applied_compensation);
        self::assertEquals($value, is_array($value) ? $instance->getAppliedCompensation()->toArray() : $instance->getAppliedCompensation());
        self::assertEquals($value, is_array($value) ? $instance->applied_compensation->toArray() : $instance->applied_compensation);
    }

    /**
    * Test invalid property "applied_compensation"
    * @dataProvider invalidAppliedCompensationDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidAppliedCompensation(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAppliedCompensation($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validAppliedCompensationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_applied_compensation'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidAppliedCompensationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_applied_compensation'));
    }
}
