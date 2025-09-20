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

use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\CaptureDealData;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель CreateCaptureRequest.
 *
 * Класс объекта запроса к API на подтверждение оплаты.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AmountInterface $amount Подтверждаемая сумма оплаты
 * @property ReceiptInterface $receipt Данные фискального чека 54-ФЗ
 * @property CaptureDealData $deal Данные о сделке, в составе которой проходит платеж
 */
class CreateCaptureRequest extends AbstractPaymentRequest implements CreateCaptureRequestInterface
{
    /**
     * @var CaptureDealData|null Данные о сделке, в составе которой проходит платеж
     */
    #[Assert\Valid]
    #[Assert\Type(CaptureDealData::class)]
    private ?CaptureDealData $_deal = null;

    /**
     * Возвращает данные о сделке, в составе которой проходит платеж.
     *
     * @return null|CaptureDealData Данные о сделке, в составе которой проходит платеж
     */
    public function getDeal(): ?CaptureDealData
    {
        return $this->_deal;
    }

    /**
     * Проверяет, были ли установлены данные о сделке.
     *
     * @return bool True если данные о сделке были установлены, false если нет
     */
    public function hasDeal(): bool
    {
        return !empty($this->_deal);
    }

    /**
     * Устанавливает данные о сделке, в составе которой проходит платеж.
     *
     * @param null|array|CaptureDealData $deal Данные о сделке, в составе которой проходит платеж
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданные данные не удалось интерпретировать как метаданные платежа
     */
    public function setDeal(mixed $deal): self
    {
        $this->_deal = $this->validatePropertyValue('_deal', $deal);
        return $this;
    }

    /**
     * Валидирует объект запроса.
     *
     * @return bool True если запрос валиден и его можно отправить в API, false если нет
     */
    public function validate(): bool
    {
        if ($this->hasAmount()) {
            $value = $this->getAmount()?->getValue();
            if (empty($value) || $value <= 0.0) {
                $this->setValidationError('Invalid amount value: ' . $value);

                return false;
            }
        }
        if ($this->hasReceipt() && $this->getReceipt()?->notEmpty()) {
            $email = $this->getReceipt()?->getCustomer()?->getEmail();
            $phone = $this->getReceipt()?->getCustomer()?->getPhone();
            if (empty($email) && empty($phone)) {
                $this->setValidationError('Both email and phone values are empty in receipt');

                return false;
            }
        }

        return true;
    }

    /**
     * Возвращает билдер объектов запросов на подтверждение оплаты.
     *
     * @return CreateCaptureRequestBuilder Инстанс билдера
     */
    public static function builder(): CreateCaptureRequestBuilder
    {
        return new CreateCaptureRequestBuilder();
    }
}
