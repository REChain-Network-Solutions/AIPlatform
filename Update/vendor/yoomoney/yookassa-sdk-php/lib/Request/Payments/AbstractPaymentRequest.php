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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractRequest;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель AbstractPaymentRequest.
 *
 * Базовый класс объекта запроса к API.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AmountInterface $amount Сумма
 * @property ReceiptInterface|null $receipt Данные фискального чека 54-ФЗ
 * @property AirlineInterface|null $airline Объект с данными для продажи авиабилетов
 * @property ListObjectInterface|TransferDataInterface[]|null $transfers Данные о распределении платежа между магазинами
 */
class AbstractPaymentRequest extends AbstractRequest
{
    /**
     * @var AmountInterface|null Сумма оплаты
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    protected ?AmountInterface $_amount = null;

    /**
     * @var Receipt|null Данные фискального чека 54-ФЗ
     */
    #[Assert\Valid]
    #[Assert\Type(Receipt::class)]
    protected ?ReceiptInterface $_receipt = null;

    /**
     * @var AirlineInterface|null Объект с данными для продажи авиабилетов
     */
    #[Assert\Valid]
    #[Assert\Type(Airline::class)]
    protected ?AirlineInterface $_airline = null;

    /**
     * @var TransferDataInterface[]|ListObjectInterface|null
     */
    #[Assert\Valid]
    #[Assert\AllType(TransferData::class)]
    #[Assert\Type(ListObject::class)]
    protected ?ListObject $_transfers = null;

    /**
     * Возвращает сумму оплаты.
     *
     * @return AmountInterface|null Сумма оплаты
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Проверяет, была ли установлена сумма оплаты.
     *
     * @return bool True если сумма оплаты была установлена, false если нет
     */
    public function hasAmount(): bool
    {
        return !empty($this->_amount);
    }

    /**
     * Устанавливает сумму оплаты.
     *
     * @param AmountInterface|array|string $amount Сумма оплаты
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): AbstractRequestInterface
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает чек, если он есть.
     *
     * @return null|ReceiptInterface Данные фискального чека 54-ФЗ или null, если чека нет
     */
    public function getReceipt(): ?ReceiptInterface
    {
        return $this->_receipt;
    }

    /**
     * Устанавливает чек.
     *
     * @param null|array|ReceiptInterface $receipt Инстанс чека или null для удаления информации о чеке
     *
     * @return self
     */
    public function setReceipt(mixed $receipt): AbstractRequestInterface
    {
        $this->_receipt = $this->validatePropertyValue('_receipt', $receipt);
        return $this;
    }

    /**
     * Проверяет наличие чека.
     *
     * @return bool True если чек есть, false если нет
     */
    public function hasReceipt(): bool
    {
        return (bool) $this->getReceipt()?->notEmpty();
    }

    /**
     * Удаляет чек из запроса.
     *
     * @return self
     */
    public function removeReceipt(): AbstractRequestInterface
    {
        $this->_receipt = null;
        return $this;
    }

    /**
     * Возвращает данные авиабилетов.
     *
     * @return null|AirlineInterface Данные авиабилетов
     */
    public function getAirline(): ?AirlineInterface
    {
        return $this->_airline;
    }

    /**
     * Проверяет, были ли установлены данные авиабилетов.
     */
    public function hasAirline(): bool
    {
        return null !== $this->_airline;
    }

    /**
     * Устанавливает данные авиабилетов.
     *
     * @param AirlineInterface|array|null $airline Данные авиабилетов
     */
    public function setAirline(mixed $airline): AbstractRequestInterface
    {
        $this->_airline = $this->validatePropertyValue('_airline', $airline);
        return $this;
    }

    /**
     * Проверяет наличие данных о распределении денег.
     */
    public function hasTransfers(): bool
    {
        return !$this->getTransfers()->isEmpty();
    }

    /**
     * Возвращает данные о распределении денег — сколько и в какой магазин нужно перевести.
     * Присутствует, если вы используете решение ЮKassa для платформ.
     * (https://yookassa.ru/developers/special-solutions/checkout-for-platforms/basics).
     *
     * @return TransferDataInterface[]|ListObjectInterface Данные о распределении денег
     */
    public function getTransfers(): ListObjectInterface
    {
        if ($this->_transfers === null) {
            $this->_transfers = new ListObject(TransferData::class);
        }
        return $this->_transfers;
    }

    /**
     * Устанавливает transfers (массив распределения денег между магазинами).
     *
     * @param ListObjectInterface|array|null $transfers Массив распределения денег
     *
     * @return self
     */
    public function setTransfers(mixed $transfers = null): AbstractRequestInterface
    {
        $this->_transfers = $this->validatePropertyValue('_transfers', $transfers);
        return $this;
    }

    /**
     * Валидирует объект запроса.
     *
     * @return bool True если запрос валиден и его можно отправить в API, false если нет
     */
    public function validate(): bool
    {
        if (null === $this->_amount) {
            $this->setValidationError('Payment amount not specified');

            return false;
        }

        $value = $this->_amount->getValue();
        if (empty($value) || $value <= 0.0) {
            $this->setValidationError('Invalid Payment amount value: ' . $value);

            return false;
        }

        if ($this->hasTransfers()) {
            $sum = 0;
            foreach ($this->getTransfers() as $i => $transfer) {
                if (null === $transfer->getAmount()) {
                    $this->setValidationError('Transfer[' . $i . '] amount not specified');

                    return false;
                }

                $value = $transfer->getAmount()?->getValue();
                if (empty($value) || $value <= 0.0) {
                    $this->setValidationError('Invalid Transfer[' . $i . '] amount value: ' . $value);

                    return false;
                }
                $sum += (float) $value;

                $accountId = $transfer->getAccountId();
                if (empty($accountId)) {
                    $this->setValidationError('Transfer[' . $i . '] account id not specified');

                    return false;
                }
            }

            if ($sum !== (float) $this->getAmount()?->getValue()) {
                $this->setValidationError('Transfer amount sum does not match top-level amount');
            }
        }

        if ($this->getReceipt()?->notEmpty()) {
            $email = $this->getReceipt()?->getCustomer()?->getEmail();
            $phone = $this->getReceipt()?->getCustomer()?->getPhone();
            if (empty($email) && empty($phone)) {
                $this->setValidationError('Both email and phone values are empty in receipt');

                return false;
            }
        }

        return true;
    }
}
