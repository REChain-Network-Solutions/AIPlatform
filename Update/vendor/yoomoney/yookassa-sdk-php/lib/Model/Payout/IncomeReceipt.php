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

namespace YooKassa\Model\Payout;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель IncomeReceipt.
 *
 * Данные чека, зарегистрированного в ФНС.
 * Присутствует, если вы делаете выплату [самозанятому](/developers/payouts/scenario-extensions/self-employed).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $service_name Описание услуги, оказанной получателем выплаты. Не более 50 символов.
 * @property string $serviceName Описание услуги, оказанной получателем выплаты. Не более 50 символов.
 * @property string $npd_receipt_id Идентификатор чека в сервисе.
 * @property string $npdReceiptId Идентификатор чека в сервисе.
 * @property string $url Ссылка на зарегистрированный чек.
 * @property AmountInterface $amount Сумма, указанная в чеке. Присутствует, если в запросе передавалась сумма для печати в чеке.
 */
class IncomeReceipt extends AbstractObject
{
    /** @var int Максимальная длина описание услуги */
    public const MAX_LENGTH_SERVICE_NAME = 50;

    /** @var int Максимальная длина описание услуги */
    public const MIN_LENGTH_SERVICE_NAME = 1;

    /**
     * Описание услуги, оказанной получателем выплаты. Не более 50 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_SERVICE_NAME)]
    #[Assert\Length(min: self::MIN_LENGTH_SERVICE_NAME)]
    private ?string $_service_name = null;

    /**
     * Идентификатор чека в сервисе. Пример: ~`208jd98zqe`
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_npd_receipt_id = null;

    /**
     * Ссылка на зарегистрированный чек.
     * Пример: ~`https://www.nalog.gov.ru/api/v1/receipt/<Идентификатор чека>/print`
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_url = null;

    /**
     * Сумма, указанная в чеке. Присутствует, если в запросе передавалась сумма для печати в чеке.
     *
     * @var AmountInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Возвращает service_name.
     *
     * @return string|null
     */
    public function getServiceName(): ?string
    {
        return $this->_service_name;
    }

    /**
     * Устанавливает service_name.
     *
     * @param string|null $service_name Описание услуги, оказанной получателем выплаты. Не более 50 символов.
     *
     * @return self
     */
    public function setServiceName(?string $service_name = null): self
    {
        $this->_service_name = $this->validatePropertyValue('_service_name', $service_name);
        return $this;
    }

    /**
     * Возвращает npd_receipt_id.
     *
     * @return string|null
     */
    public function getNpdReceiptId(): ?string
    {
        return $this->_npd_receipt_id;
    }

    /**
     * Устанавливает npd_receipt_id.
     *
     * @param string|null $npd_receipt_id Идентификатор чека в сервисе
     *
     * @return self
     */
    public function setNpdReceiptId(?string $npd_receipt_id = null): self
    {
        $this->_npd_receipt_id = $this->validatePropertyValue('_npd_receipt_id', $npd_receipt_id);
        return $this;
    }

    /**
     * Возвращает ссылку на зарегистрированный чек.
     *
     * @return string|null
     */
    public function getUrl(): ?string
    {
        return $this->_url;
    }

    /**
     * Устанавливает ссылку на зарегистрированный чек.
     *
     * @param string|null $url Ссылка на зарегистрированный чек
     *
     * @return self
     */
    public function setUrl(?string $url = null): self
    {
        $this->_url = $this->validatePropertyValue('_url', $url);
        return $this;
    }

    /**
     * Возвращает amount.
     *
     * @return AmountInterface|null
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает amount.
     *
     * @param AmountInterface|array|null $amount
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }
}
