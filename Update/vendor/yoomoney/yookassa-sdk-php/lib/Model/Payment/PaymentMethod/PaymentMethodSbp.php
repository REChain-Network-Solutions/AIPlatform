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

namespace YooKassa\Model\Payment\PaymentMethod;

use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodSbp.
 *
 * Оплата через СБП (Система быстрых платежей ЦБ РФ).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string|null $sbp_operation_id Идентификатор операции в СБП (НСПК).
 * @property string|null $sbpOperationId Идентификатор операции в СБП (НСПК).
 * @property SbpPayerBankDetails|null $payer_bank_details Реквизиты счета, который использовался для оплаты.
 * @property SbpPayerBankDetails|null $payerBankDetails Реквизиты счета, который использовался для оплаты.
 * @property string|null $qrc_id Идентификатор кассовой ссылки СБП в сервисе НСПК.
 * @property string|null $qrcId Идентификатор кассовой ссылки СБП в сервисе НСПК.
 * @property string|null $params_id Идентификатор активных значений параметров Платежной ссылки СБП.
 * @property string|null $paramsId Идентификатор активных значений параметров Платежной ссылки СБП.
 */
class PaymentMethodSbp extends AbstractPaymentMethod
{
    /**
     * Идентификатор операции в СБП (НСПК). Пример: `1027088AE4CB48CB81287833347A8777`
     * Обязательный параметр для платежей в статусе ~`succeeded`. В остальных случаях может отсутствовать.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_sbp_operation_id = null;

    /**
     * Реквизиты счета, который использовался для оплаты.
     * Обязательный параметр для платежей в статусе ~`succeeded`. В остальных случаях может отсутствовать.
     *
     * @var SbpPayerBankDetails|null
     */
    #[Assert\Type(SbpPayerBankDetails::class)]
    private ?SbpPayerBankDetails $_payer_bank_details = null;

    /**
     * Идентификатор кассовой ссылки СБП в сервисе НСПК. Пример: `AS1F001K6S85322B9E4P7I518N8LNEUT`
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(min: 32)]
    #[Assert\Length(max: 32)]
    #[Assert\Regex("/^[A-Za-z0-9]{32}$/")]
    private ?string $_qrc_id = null;

    /**
     * Идентификатор активных значений параметров Платежной ссылки СБП. Присутствует, если оплата происходила по Кассовой ссылке СБП. Пример: `AP5GK3L276DN8PDR8ON1LF4RVOL44444`
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(min: 32)]
    #[Assert\Length(max: 32)]
    #[Assert\Regex("/^[A-Za-z0-9]{32}$/")]
    private ?string $_params_id = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::SBP);
    }

    /**
     * Возвращает sbp_operation_id.
     *
     * @return string|null Идентификатор операции в СБП (НСПК).
     */
    public function getSbpOperationId(): ?string
    {
        return $this->_sbp_operation_id;
    }

    /**
     * Устанавливает sbp_operation_id.
     *
     * @param string|null $sbp_operation_id Идентификатор операции в СБП (НСПК).
     *
     * @return self
     */
    public function setSbpOperationId(?string $sbp_operation_id = null): self
    {
        $this->_sbp_operation_id = $this->validatePropertyValue('_sbp_operation_id', $sbp_operation_id);
        return $this;
    }

    /**
     * Возвращает payer_bank_details.
     *
     * @return SbpPayerBankDetails|null Реквизиты счета, который использовался для оплаты.
     */
    public function getPayerBankDetails(): ?SbpPayerBankDetails
    {
        return $this->_payer_bank_details;
    }

    /**
     * Устанавливает payer_bank_details.
     *
     * @param SbpPayerBankDetails|array|null $payer_bank_details Реквизиты счета, который использовался для оплаты.
     *
     * @return self
     */
    public function setPayerBankDetails(mixed $payer_bank_details = null): self
    {
        $this->_payer_bank_details = $this->validatePropertyValue('_payer_bank_details', $payer_bank_details);
        return $this;
    }

    /**
     * Возвращает идентификатор кассовой ссылки СБП в сервисе НСПК
     *
     * @return string|null
     */
    public function getQrcId(): ?string
    {
        return $this->_qrc_id;
    }

    /**
     * Устанавливает идентификатор кассовой ссылки СБП в сервисе НСПК
     *
     * @param string|null $qrcId Идентификатор кассовой ссылки СБП в сервисе НСПК
     *
     * @return self
     */
    public function setQrcId(?string $qrcId = null): self
    {
        $this->_qrc_id = $this->validatePropertyValue('_qrc_id', $qrcId);
        return $this;
    }

    /**
     * Возвращает идентификатор активных значений параметров Платежной ссылки СБП
     *
     * @return string|null
     */
    public function getParamsId(): ?string
    {
        return $this->_params_id;
    }

    /**
     * Устанавливает идентификатор активных значений параметров Платежной ссылки СБП
     *
     * @param string|null $paramsId Идентификатор активных значений параметров Платежной ссылки СБП
     *
     * @return self
     */
    public function setParamsId(?string $paramsId = null): self
    {
        $this->_params_id = $this->validatePropertyValue('_params_id', $paramsId);
        return $this;
    }
}
