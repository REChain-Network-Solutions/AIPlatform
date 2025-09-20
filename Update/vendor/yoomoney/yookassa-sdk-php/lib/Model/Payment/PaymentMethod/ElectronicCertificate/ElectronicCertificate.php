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

namespace YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ElectronicCertificate.
 *
 * Описание используемого электронного сертификата.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $certificate_id Идентификатор сертификата. От 20 до 30 символов.
 * @property string $certificateId Идентификатор сертификата. От 20 до 30 символов.
 * @property int $tru_quantity Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату.
 * @property int $truQuantity Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату.
 * @property AmountInterface $available_compensation Максимально допустимая сумма, которую может покрыть электронный сертификат для оплаты одной единицы товара. Пример: сертификат может компенсировать максимум 1000 рублей для оплаты этого товара.
 * @property AmountInterface $availableCompensation Максимально допустимая сумма, которую может покрыть электронный сертификат для оплаты одной единицы товара. Пример: сертификат может компенсировать максимум 1000 рублей для оплаты этого товара.
 * @property AmountInterface $applied_compensation Сумма, которую одобрили для оплаты по сертификату за одну единицу товара. Пример: из 1000 рублей одобрили 500 рублей для оплаты по сертификату.
 * @property AmountInterface $appliedCompensation Сумма, которую одобрили для оплаты по сертификату за одну единицу товара. Пример: из 1000 рублей одобрили 500 рублей для оплаты по сертификату.
 */
class ElectronicCertificate extends AbstractObject
{
    /**
     * Идентификатор сертификата. От 20 до 30 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 30)]
    #[Assert\Length(min: 20)]
    private ?string $_certificate_id = null;

    /**
     * Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    private ?int $_tru_quantity = null;

    /**
     * Максимально допустимая сумма, которую может покрыть электронный сертификат для оплаты одной единицы товара. Пример: сертификат может компенсировать максимум 1000 рублей для оплаты этого товара.
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_available_compensation = null;

    /**
     * Сумма, которую одобрили для оплаты по сертификату за одну единицу товара. Пример: из 1000 рублей одобрили 500 рублей для оплаты по сертификату.
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_applied_compensation = null;

    /**
     * Возвращает certificate_id.
     *
     * @return string|null Идентификатор сертификата
     */
    public function getCertificateId(): ?string
    {
        return $this->_certificate_id;
    }

    /**
     * Устанавливает certificate_id.
     *
     * @param string|null $certificate_id Идентификатор сертификата. От 20 до 30 символов.
     *
     * @return self
     */
    public function setCertificateId(?string $certificate_id = null): self
    {
        $this->_certificate_id = $this->validatePropertyValue('_certificate_id', $certificate_id);
        return $this;
    }

    /**
     * Возвращает tru_quantity.
     *
     * @return int|null Количество единиц товара
     */
    public function getTruQuantity(): ?int
    {
        return $this->_tru_quantity;
    }

    /**
     * Устанавливает tru_quantity.
     *
     * @param int|null $tru_quantity Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату.
     *
     * @return self
     */
    public function setTruQuantity(?int $tru_quantity = null): self
    {
        $this->_tru_quantity = $this->validatePropertyValue('_tru_quantity', $tru_quantity);
        return $this;
    }

    /**
     * Возвращает available_compensation.
     *
     * @return AmountInterface|null Максимально допустимая сумма
     */
    public function getAvailableCompensation(): ?AmountInterface
    {
        return $this->_available_compensation;
    }

    /**
     * Устанавливает available_compensation.
     *
     * @param AmountInterface|array|null $available_compensation Максимально допустимая сумма
     *
     * @return self
     */
    public function setAvailableCompensation(mixed $available_compensation = null): self
    {
        $this->_available_compensation = $this->validatePropertyValue('_available_compensation', $available_compensation);
        return $this;
    }

    /**
     * Возвращает applied_compensation.
     *
     * @return AmountInterface|null Сумма, которую одобрили для оплаты по сертификату
     */
    public function getAppliedCompensation(): ?AmountInterface
    {
        return $this->_applied_compensation;
    }

    /**
     * Устанавливает applied_compensation.
     *
     * @param AmountInterface|array|null $applied_compensation Сумма, которую одобрили для оплаты по сертификату
     *
     * @return self
     */
    public function setAppliedCompensation(mixed $applied_compensation = null): self
    {
        $this->_applied_compensation = $this->validatePropertyValue('_applied_compensation', $applied_compensation);
        return $this;
    }

}

